# Extracts bias and non-bias parameters from a model.
def get_parameters(model, bias=False):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None


# from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


# Convert to Long Tensor from Byte Tensor
class toLongTensor:
    def __call__(self, img):
        output = torch.from_numpy(np.array(img).astype(np.int32)).long()
        output[output == 255] = 21
        return output


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """ Returns overall accuracy and mean IoU """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iou = np.nanmean(iou)
    return acc, mean_iou


class Colorize(object):
    """ Colorize the segmentation labels """
    def __init__(self, n=35, cmap=None):
        if cmap is None:
            raise NotImplementedError()
            self.cmap = labelcolormap(n)
        else:
            self.cmap = cmap
        self.cmap = self.cmap[:n]
    def preprocess(self, x):
        if len(x.size()) > 3 and x.size(1) > 1:
            # if x has a shape of [B, C, H, W],
            # where B and C denote a batch size and the number of semantic classe
            # then translate it into a shape of [B, 1, H, W]
            x = x.argmax(dim=1, keepdim=True).float()
        assert (len(x.shape) == 4) and (x.size(1) == 1), 'x should have a shape of [B, 1, H, W]'
        return x
    def __call__(self, x):
        x = self.preprocess(x)
        if (x.dtype == torch.float) and (x.max() < 2):
            x = x.mul(255).long()
        color_images = []
        gray_image_shape = x.shape[1:]
        for gray_image in x:
            color_image = torch.ByteTensor(3, *gray_image_shape[1:]).fill_(0)
            for label, cmap in enumerate(self.cmap):
                mask = (label == gray_image[0]).cpu()
                color_image[0][mask] = cmap[0]
                color_image[1][mask] = cmap[1]
                color_image[2][mask] = cmap[2]
            color_images.append(color_image)
        color_images = torch.stack(color_images)
        return color_images


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def get_color_map():
    """ returns N color map """
    N=25
    color_map = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        color_map[i, 0] = r
        color_map[i, 1] = g
        color_map[i, 2] = b
    color_map = torch.from_numpy(color_map)
    return color_map


# Conditional Random Field for better segmentation
# Refer to https://github.com/lucasb-eyer/pydensecrf for details.
def dense_crf(img, output_probs):
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=img, compat=4)

    Q = d.inference(10)
    Q = np.array(Q).reshape((c, h, w))
    return Q


# Zero-pad image(or any array-like object) to 500x500.
def add_padding(img):
    w, h = img.shape[-2], img.shape[-1]
    MAX_SIZE = 500
    IGNORE_IDX = 21

    assert max(w, h) <= MAX_SIZE, f'both height and width should be less than {MAX_SIZE}'

    _pad_left = (MAX_SIZE - w) // 2
    _pad_right = (MAX_SIZE - w + 1) // 2
    _pad_up = (MAX_SIZE - h) // 2
    _pad_down = (MAX_SIZE - h + 1) // 2

    _pad = (_pad_up, _pad_down, _pad_left, _pad_right)

    padding_img = transforms.Pad(_pad)
    padding_target = transforms.Pad(_pad, fill=IGNORE_IDX)

    img = F.pad(img, pad=_pad)
    return img

#------------Implementing FCN32-----------------#

class FCN32(VGG):
    def __init__(self):
        super(FCN32, self).__init__(make_layers(cfg['vgg16']))

        self.numclass = 21

        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d()

        # fc layers in vgg are all converted into conv layers.
        #################################
        self.conv1 = nn.Conv2d(512,4096,kernel_size = 7)
        #relu + # dropout 적용
        self.conv2 = nn.Conv2d(4096,4096,kernel_size = 1)
        #relu + # dropout 적용
        #################################

        # Prediction layer with 1x1 convolution layer.
        #################################
        self.conv3 = nn.Conv2d(4096,self.numclass,kernel_size = 1)
        #################################

        # Learnable upsampling layers in FCN model.
        #################################
        self.convtrans = nn.ConvTranspose2d(self.numclass ,self.numclass ,kernel_size = 64, stride = 32 , bias = False)
        #################################

        # initialize deconv layer with bilinear upsampling.
        self._initialize_weights()

    def load_pretrained(self, pretrained_model):
        self.features = pretrained_model.features
        fc6 = pretrained_model.classifier[0]
        fc7 = pretrained_model.classifier[3]

        # print(self.conv1.weight.data.size())
        # print(fc6.weight.data.size())
        #################################
        self.conv1.weight.data = fc6.weight.data.view(self.conv1.weight.size())
        self.conv2.weight.data = fc7.weight.data.view(self.conv2.weight.size())
        #################################

    def vgg_layer_forward(self, x, indices):
        output = x
        start_idx, end_idx = indices
        for idx in range(start_idx, end_idx):
            output = self.features[idx](output)
        return output

    def vgg_forward(self, x):
        out = {}
        layer_indices = [0, 5, 10, 17, 24, 31]
        for layer_num in range(len(layer_indices)-1):
            x = self.vgg_layer_forward(x, layer_indices[layer_num:layer_num+2])
            out[f'pool{layer_num+1}'] = x
        return out

    def forward(self, x):
        # Padding for aligning to the input size
        padded_x = F.pad(x, [100, 100, 100, 100], "constant", 0)
        vgg_features = self.vgg_forward(padded_x)
        vgg_pool5 = vgg_features['pool5'].detach()
        vgg_pool4 = vgg_features['pool4'].detach()
        vgg_pool3 = vgg_features['pool3'].detach()

        #################################
        out = (self.convtrans(self.conv3(self.dropout(self.relu(self.conv2(self.dropout(self.relu(self.conv1(vgg_pool5)))))))))
        out = transforms.functional.crop(out, left=9, top=9, height=x.size(2), width=x.size(3))
        
        return out

        #################################

        

    # initialize transdeconv layer with bilinear upsampling.
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

#------------Implementing FCN8-----------------#

class FCN8(FCN32):
    def __init__(self):
        super(FCN8, self).__init__()

        self.numclass = 21

        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d()

        # fc layers in vgg are all converted into conv layers.
        #################################
        self.conv1 = nn.Conv2d(512, 4096,kernel_size = 7)
        self.conv2 = nn.Conv2d(4096, 4096,kernel_size = 1)
        #################################

        # prediction layers with 1x1 convolution layers.
        #################################
        self.conv3 = nn.Conv2d(4096,self.numclass,kernel_size = 1)
        self.conv4 = nn.Conv2d(512,self.numclass,kernel_size = 1)
        self.conv5 = nn.Conv2d(256,self.numclass,kernel_size = 1)
        #################################

        # Learnable upsampling layers in FCN model.
        #################################
        self.convtrans1 = nn.ConvTranspose2d(self.numclass,out_channels = self.numclass, kernel_size =4 , stride = 2, bias = False)
        self.convtrans2 = nn.ConvTranspose2d(self.numclass,out_channels = self.numclass, kernel_size =4 , stride = 2, bias = False)
        self.convtrans3 = nn.ConvTranspose2d(self.numclass,out_channels = self.numclass, kernel_size =16 , stride = 8, bias = False)
        #################################

        # initialize deconv layer with bilinear upsampling.
        self._initialize_weights()

    def forward(self, x):
        # Padding for aligning to the input size
        padded_x = F.pad(x, [100, 100, 100, 100], "constant", 0)
        vgg_features = self.vgg_forward(padded_x)
        vgg_pool5 = vgg_features['pool5'].detach()
        vgg_pool4 = vgg_features['pool4'].detach()
        vgg_pool3 = vgg_features['pool3'].detach()

        #################################
        out1 = self.convtrans1(self.conv3(self.dropout(self.relu(self.conv2(self.dropout(self.relu(self.conv1(vgg_pool5))))))))
        
        out2 = self.conv4(vgg_pool4 *0.01)
        out2 = transforms.functional.crop(out2, left=5, top=5, height=out1.size(2), width=out1.size(3))
        out2 = self.convtrans2(out1 + out2) # 2 번째 convtrnas
        
        out3 = self.conv5(vgg_pool3 *0.0001) # vggpol3
        out3 = transforms.functional.crop(out3, left=9, top=9, height=out2.size(2), width=out2.size(3)) # -- picture3

        out = self.convtrans3(out3+out2) # last convtrans
        out = transforms.functional.crop(out,left=31,top=31,height=x.size(2),width=x.size(3)) # last crop
        
        return out
        #################################

        

#------------Training function-----------------#

def train_net(net, resume=False):
    # 21 is the index for boundaries: therefore we ignore this index.
    criterion = nn.CrossEntropyLoss(ignore_index=21)
    colorize = Colorize(21, get_color_map())
    best_valid_iou = 0

    if resume:
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_valid_iou = checkpoint['best_valid_iou']
        print(f'Resume training from epoch {epoch+1}')
    else:
        epoch = 0

    while epoch < args.epoch:
        t1 = time.time()
        saved_images, saved_labels = [], []

        # Start training
        net.train()

        loss_total = 0
        ious = []
        pixel_accs = []

        for batch_idx, (image, label) in enumerate(train_loader):
            # Save images for visualization.
            if len(saved_images) < 4:
                saved_images.append(image.cpu())
                saved_labels.append(add_padding(label.cpu()))

            # Move variables to gpu.
            image = image.to(device)
            label = label.to(device)

            # Feed foward.
            # Calculate loss.
            # Perform gradient descent step.
            # pred: choose the label with the highest logit in each pixel.
            #################################
            ## P3(a). Write your code here ##
            output = net(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output, dim=1) # (B, H, W)
            ## Add more lines as you wish. ##
            #################################

            target = label.squeeze(1).cpu().numpy()

            # calculate pixel accuarcy and mean IoU
            acc, mean_iu = label_accuracy_score(target, pred.cpu().detach().numpy(), n_class=21)

            pixel_accs.append(acc)
            ious.append(mean_iu)

            # Update total loss.
            loss_total += loss.item()

            if batch_idx % 50 == 0:
                print(f'Epoch : {epoch} || {batch_idx}/{len(train_loader)} || \
                 loss : {loss.item():.3f}, iou : {mean_iu * 100:.3f} pixel_acc : {acc * 100:.3f}')

        # Calculate average IoU
        total_ious = np.array(ious).T
        total_ious = np.nanmean(total_ious).mean()
        total_pixel_acc = np.array(pixel_accs).mean()

        writer.add_scalar('train_loss', loss_total / len(train_loader), epoch)
        writer.add_scalar('pixel_acc', total_pixel_acc, epoch)
        writer.add_scalar('mean_iou', total_ious, epoch)

        # Image visualization
        un_norms, preds, outputs = [], [], []
        for image, label in zip(saved_images, saved_labels):
            # Denormalize the image.
            image_permuted = image.permute(1, 0, 2, 3)
            un_norm = torch.zeros_like(image_permuted)
            for idx, (im, m, s) in enumerate(zip(image_permuted, mean, std)):
                un_norm[idx] = (im * s) + m
            un_norm = un_norm.permute(1, 0, 2, 3)
            un_norms.append(add_padding(un_norm))

            with torch.no_grad():
                output = net(image.to(device))
                pred = torch.argmax(output, dim=1)
                preds.append(add_padding(pred))
                
        # Stitch images into a grid.
        un_norm = make_grid(torch.cat(un_norms), nrow=2)
        label = make_grid(colorize(torch.stack(saved_labels)), nrow=2)
        pred = make_grid(colorize(torch.stack(preds)), nrow=2)

        # Write images to Tensorboard.
        writer.add_image('img', un_norm, epoch)
        writer.add_image('gt', label, epoch)
        writer.add_image('pred', pred, epoch)

        t = time.time() - t1
        print(f'>> Epoch : {epoch} || AVG loss : {loss_total / len(train_loader):.3f}, \
         iou : {total_ious * 100:.3f} pixel_acc : {total_pixel_acc * 100:.3f} {t:.3f} secs')

        # Evaluation
        net.eval()
        saved_images, saved_labels = [], []

        valid_loss_total = 0
        valid_ious = []
        valid_pixel_accs = []

        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(test_loader):
                # Save images for visualization.
                if len(saved_images) < 4:
                    saved_images.append(image.cpu())
                    saved_labels.append(add_padding(label.cpu()))
                
                # Move variables to gpu.
                image = image.to(device)
                label = label.to(device)

                # Feed foward.
                # Calculate loss.
                # pred: choose the label with highest logit in each pixel.
                #################################
                ## P3(b). Write your code here ##
                ## Add more lines as you wish. ##
                output = net(image)
                loss = criterion(output, label)
                pred = torch.argmax(output, dim=1)
                #################################

                target = label.squeeze(1).cpu().numpy()
                acc, mean_iu = label_accuracy_score(target, pred.cpu().numpy(), n_class=21)

                # Update total loss.
                valid_loss_total += loss.item()

                valid_pixel_accs.append(acc)
                valid_ious.append(mean_iu)

        # Calculate average IoU
        total_valid_ious = np.array(valid_ious).T
        total_valid_ious = np.nanmean(total_valid_ious).mean()
        total_valid_pixel_acc = np.array(valid_pixel_accs).mean()

        writer.add_scalar('valid_train_loss', valid_loss_total / len(test_loader), epoch)
        writer.add_scalar('valid_pixel_acc', total_valid_pixel_acc, epoch)
        writer.add_scalar('valid_mean_iou', total_valid_ious, epoch)

        # Image visualization + CRF 
        un_norms, preds, pred_crfs, outputs = [], [], [], []
        for image, label in zip(saved_images, saved_labels):
            # Denormalize the image.
            image_permuted = image.permute(1, 0, 2, 3)
            un_norm = torch.zeros_like(image_permuted)
            for idx, (im, m, s) in enumerate(zip(image_permuted, mean, std)):
                un_norm[idx] = (im * s) + m
            un_norm = un_norm.permute(1, 0, 2, 3)
            un_norms.append(add_padding(un_norm))

            with torch.no_grad():
                output = net(image.to(device))
                outputs.append(add_padding(output))
                pred = torch.argmax(output, dim=1)
                preds.append(add_padding(pred))

            # CRF
            output_softmax = torch.nn.functional.softmax(output, dim=1).detach().cpu()
            un_norm_int = (un_norm * 255).squeeze().permute(1, 2, 0).numpy().astype(np.ubyte)
            pred_crf = dense_crf(un_norm_int, output_softmax.squeeze().numpy())
            pred_crfs.append(add_padding(torch.argmax(torch.Tensor(pred_crf), dim=0)).unsqueeze(0))

        # Stitch images into a grid.
        valid_un_norm = make_grid(torch.cat(un_norms), nrow=2)
        valid_label = make_grid(colorize(torch.stack(saved_labels)), nrow=2)
        valid_pred = make_grid(colorize(torch.stack(preds)), nrow=2)
        valid_pred_crf = make_grid(colorize(torch.stack(pred_crfs)), nrow=2)

        # Write images to tensorboard.
        writer.add_image('valid_img', valid_un_norm, epoch)
        writer.add_image('valid_gt', valid_label, epoch)
        writer.add_image('valid_pred', valid_pred, epoch)
        writer.add_image('valid_pred_crf', valid_pred_crf, epoch)

        print(f'>> Epoch : {epoch} || AVG valid loss : {valid_loss_total / len(test_loader):.3f}, \
          iou : {total_valid_ious * 100:.3f} pixel_acc : {total_valid_pixel_acc * 100:.3f} {t:.3f} secs')

        # Save checkpoints every epoch.
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_valid_iou': best_valid_iou
        }
        torch.save(checkpoint, ckpt_path)

        # Save best checkpoint.
        if total_valid_ious > best_valid_iou:
            best_valid_iou = total_valid_ious
            torch.save(net.state_dict(), ckpt_dir / 'best.pt')

        epoch += 1
    print(f'>> Best validation set iou: {best_valid_iou}')