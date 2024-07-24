```
#---------------------------相对的改进，如需训练先执行annotation.py划分出本地的图片集合，纯推理预测则不用------------------------#
#---------------------------(predict.ipynb)------------------------#
    if mode == "predict":
        image = Image.open('4.jpg')#先自动对目录下的4.jpg文件实施基线预测
        #此外还提供的基准图片有：45是基线目标，67是大目标，89是小目标，cd是难目标
        r_image = yolo.detect_image(image, crop = crop, count=count)
        r_image.show()
        while True:
            img = input('Input image filename:')
            try:#这样自动叠加后缀就只需要输入文件名
                image = Image.open(img+'.jpg')
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()
#---------------------------(yolo.py)------------------------#
        "model_path"        : 'model_data/b基础633.pth',#原yolov8_s换为自训的基线权
        "classes_path"      : 'model_data/voc_classes.txt',#只含0到6七类，分别分行
        "phi"               : 'n',#版本从s换为更易训、内存更小的n 
        "cuda"              : False,#cuda换为否方便推理时切无卡模式用cpu更省钱
#---------------------------(utils_fit.py)------------------------#
    if local_rank == 0:#去掉开训和完训，以及验证全程的显示
        # print('Start Train')
    if local_rank == 0:
        pbar.close()
        # print('Finish Train')
        # print('Start Validation')
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    if local_rank == 0:
        pbar.close()
        # print('Finish Validation')
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):#关掉最优权的保存提示，将定期权重名改为p030三个数的形式，忽略具体损失，最后精简best_epoch_weights为b，last_epoch_weights为l
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            # print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "b.pth"))
        #     torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        # torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "l.pth"))
#---------------------------(callbacks.py)------------------------#
            # print("Calculate Map.")
            # print("Get map done.") #关掉算map始末的提示
#---------------------------(train.ipynb)------------------------#
if __name__ == "__main__": #精简参数行，去除多余注释
    Cuda            = True #服务器训练只能用gpu，无卡模式cpu训不了
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True #设true更快些
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'b基础633.pth' #原为'model_data/yolov8_s.pth'改成咱们自训的
    input_shape     = [640, 640]
    phi             = 'n' # 原's'改更小更高效
    pretrained      = False #有权重就不用预训练
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    label_smoothing     = 0
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2 #原32改小
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 4 #原16改小
    Freeze_Train        = False #预冻结前50的骨网权重，在前置网需要同时训练故设False
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = "cos"
    save_period         = 30 #每隔30轮保存下权重，整个只需10个文件，减少原10的冗余
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 10
    num_workers         = 4
```
骨网部分
```
#------------------------(backbone.py)-----------------#
class Tb(nn.Module):#五卷
    def __init__(self, c):
        super(Tb, self).__init__()
        self.头=Conv(c,c,3,1)
        self.中=nn.Sequential(Conv(2*c,c),Conv(c,c,3,g=c))
        self.末=nn.Sequential(Conv(3*c,c),Conv(c,c,3,g=c))
        self.尾=nn.Sequential(Conv(4*c,c),Conv(c,c,3,g=c))
        self.终=nn.Sequential(Conv(5*c,c),Conv(c,c,3,g=c))
        self.出=Conv(5*c,2*c,1,2)
        self.池=nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
    def forward(self, x):
        x1=self.头(x);y=torch.cat([x1,self.池(x)],1)
        x2=self.中(y);z=torch.cat([self.池(y),x2],1)
        x3=self.末(z);i=torch.cat([self.池(z),x3],1)
        x4=self.尾(i);x5=self.终(torch.cat([self.池(i),x4],1))
        return self.出(torch.cat([x1,x2,x3,x4,x5],1))
class Backbone(nn.Module):
        self.dark2 = Tb(16); self.dark3 = Tb(32)
        self.dark4 = Tb(64); self.dark5 = Tb(128) #原来的代码群注释掉

#------------------------(yolo.py)-----------------#
        "model_path"        : '出简701.pth',#自训的

#------------------------(train.ipynb)-----------------#
        "model_path"        : '出简701.pth',#自训的

#--------------------(gradcam.ipynb)--------------#
        #生成图片（默认是voc文件夹中的4.jpg）经网络某显著层时的梯度热力图于result.png
```
其他还可以尝试的改进：
```
# class Tb1(nn.Module):#四内
#     def __init__(self, c):
#         super(Tb, self).__init__()
#         self.头=Conv(c,c,3,1)
#         self.中=Conv(2*c,c,3,1)
#         self.末=Conv(3*c,c,3,1)
#         self.尾=Conv(4*c,c,3,1)
#         self.出=Conv(2*c,2*c,3,2)
#         self.池=nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
#     def forward(self, x):
#         x1=self.头(x)
#         y=torch.cat([x1,self.池(x)],1)
#         x2=self.中(y)
#         z=torch.cat([self.池(y),x2],1)
#         x3=self.末(z)
#         x4=self.尾(torch.cat([self.池(z),x3],1))
#         return self.出(torch.cat([x1,x4],1))
# class Tb1(nn.Module):#加四,原为18.6,现15.7
#     def __init__(self, c):
#         super(Tb, self).__init__()
#         self.头=Conv(c,c,3,1)
#         self.中=Conv(c,c,3,1)
#         self.末=Conv(c,c,3,1)
#         self.尾=Conv(c,c,3,1)
#         self.出=Conv(4*c,2*c,3,2)
#         self.池=nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
#     def forward(self, x):
#         x1=self.头(x)
#         y=x1+self.池(x)
#         x2=self.中(y)
#         z=x2+self.池(y)
#         x3=self.末(z)
#         x4=self.尾(x3+self.池(z))
#         return self.出(torch.cat([x1,x2,x3,x4],1))
# class Tb(nn.Module):#五卷
#     def __init__(self, c):
#         super(Tb, self).__init__()
#         self.头=Conv(c,c,3,1)
#         self.中=Conv(2*c,c,3,1)
#         self.末=Conv(3*c,c,3,1)
#         self.尾=Conv(4*c,c,3,1)
#         self.终=Conv(5*c,c,3,1)
#         self.出=Conv(5*c,2*c,3,2)
#         self.池=nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
#     def forward(self, x):
#         x1=self.头(x)
#         y=torch.cat([x1,self.池(x)],1)
#         x2=self.中(y)
#         z=torch.cat([self.池(y),x2],1)
#         x3=self.末(z)
#         i=torch.cat([self.池(z),x3],1)
#         x4=self.尾(i)
#         x5=self.终(torch.cat([self.池(i),x4],1))
#         return self.出(torch.cat([x1,x2,x3,x4,x5],1))
# class Tb1(nn.Module):#七卷
#     def __init__(self, c):
#         super(Tb, self).__init__()
#         self.头=Conv(c,c,3,1)
#         self.中=Conv(2*c,c,3,1)
#         self.末=Conv(3*c,c,3,1)
#         self.尾=Conv(4*c,c,3,1)
#         self.终=Conv(5*c,c,3,1)
#         self.完=Conv(6*c,c,3,1)
#         self.出=Conv(6*c,2*c,3,2)
#         self.池=nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
#     def forward(self, x):
#         x1=self.头(x)
#         y=torch.cat([x1,self.池(x)],1)
#         x2=self.中(y)
#         z=torch.cat([self.池(y),x2],1)
#         x3=self.末(z)
#         i=torch.cat([self.池(z),x3],1)
#         x4=self.尾(i)
#         j=torch.cat([self.池(i),x4],1)
#         x5=self.终(j)
#         x6=self.完(torch.cat([self.池(j),x5],1))
#         return self.出(torch.cat([x1,x2,x3,x4,x5,x6],1))
# class Tb1(nn.Module):#独四
#     def __init__(self, c):
#         super(Tb, self).__init__()
#         self.头=Conv(c,c,3,1)
#         self.中=Conv(2*c,c,3,1)
#         self.末=Conv(3*c,c,3,1)
#         self.尾=Conv(4*c,2*c,3,2)
#         self.池=nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
#     def forward(self, x):
#         x1=self.头(x)
#         y=torch.cat([x1,self.池(x)],1)
#         x2=self.中(y)
#         z=torch.cat([self.池(y),x2],1)
#         x3=self.末(z)
#         x4=self.尾(torch.cat([self.池(z),x3],1))
#         return x4
# class Tb1(nn.Module):#外池
#     def __init__(self, c):
#         super(Tb, self).__init__()
#         self.头=Conv(c,c,3,1)
#         self.中=Conv(2*c,c,3,1)
#         self.末=Conv(3*c,c,3,1)
#         self.出=Conv(3*c,2*c,3,2)
#         self.小池=nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
#         self.大池=nn.MaxPool2d(kernel_size=9,stride=1,padding=4)
#     def forward(self, x):
#         x1=self.头(x)
#         y=torch.cat([x1,x],1)
#         x2=self.中(y)
#         x3=self.末(torch.cat([y,x2],1))
#         return self.出(torch.cat([self.小池(x1),self.大池(x2),x3],1))
# class Tb1(nn.Module):#均池
#     def __init__(self, c):
#         super(Tb, self).__init__()
#         self.头=Conv(c,c,3,1)
#         self.中=Conv(2*c,c,3,1)
#         self.末=Conv(3*c,c,3,1)
#         self.出=Conv(3*c,2*c,3,2)
#         self.池=nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
#     def forward(self, x):
#         x1=self.头(x)
#         y=torch.cat([x1,self.池(x)],1)
#         x2=self.中(y)
#         x3=self.末(torch.cat([self.池(y),x2],1))
#         return self.出(torch.cat([self.池(x1),self.池(x2),x3],1))
# class Tb1(nn.Module):#不池
#     def __init__(self, c):
#         super(Tb, self).__init__()
#         self.头=Conv(c,c,3,1)
#         self.中=Conv(2*c,c,3,1)
#         self.末=Conv(3*c,c,3,1)
#         self.出=Conv(3*c,2*c,3,2)
#     def forward(self, x):
#         x1=self.头(x)
#         y=torch.cat([x1,x],1)
#         x2=self.中(y)
#         x3=self.末(torch.cat([y,x2],1))
#         return self.出(torch.cat([x1,x2,x3],1))
# class Tb(nn.Module):#四卷
#     def __init__(self, c):
#         super(Tb, self).__init__()
#         self.头=Conv(c,c,3,1)
#         self.中=Conv(2*c,c,3,1)
#         self.末=Conv(3*c,c,3,1)
#         self.尾=Conv(4*c,c,3,1)
#         self.出=Conv(4*c,2*c,3,2)
#     def forward(self, x):
#         x1=self.头(x)
#         y=torch.cat([x1,x],1)
#         x2=self.中(y)
#         z=torch.cat([y,x2],1)
#         x3=self.末(z)
#         x4=self.尾(torch.cat([z,x3],1))
#         return self.出(torch.cat([x1,x2,x3,x4],1))
```