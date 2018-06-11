
# coding: utf-8

# In[1]:


import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import math
import random
import numpy as np
import chainer
from chainer import cuda, Chain, optimizers, Variable, serializers, Link
import chainer.functions as F
import chainer.links as L
import cupy
import cv2

import time

from loader import MultiLoader
#from cvbatch2 import CVBatch2 


# In[2]:


xp = cuda.cupy
#print(chainer.functions.Linear(1,1).type_check_enable)


# In[3]:


class EqualizedLinear(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        w = chainer.initializers.Normal(1.0) # equalized learning rate
        self.inv_c = np.sqrt(2.0/in_ch)
        super(EqualizedLinear, self).__init__()
        with self.init_scope():
            self.c = L.Linear(in_ch, out_ch, initialW=w)
            
    def __call__(self, x):
        return self.c(self.inv_c * x)


# In[4]:


class EqualizedConv2d(chainer.Chain):
    def __init__(self, in_ch, out_ch, ksize, stride, pad):
        w = chainer.initializers.Normal(1.0) # equalized learning rate
        self.inv_c = np.sqrt(2.0/(in_ch*ksize**2))
        super(EqualizedConv2d, self).__init__()
        with self.init_scope():
            self.c = L.Convolution2D(in_ch, out_ch, ksize, stride, pad, initialW=w)
            
    def __call__(self, x):
        return self.c(self.inv_c * x)


# In[5]:


class GeneratorBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        super(GeneratorBlock, self).__init__()
        self.out_ch = out_ch
        with self.init_scope():
            self.c0 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
            self.c1 = EqualizedConv2d(out_ch, out_ch, 3, 1, 1)
            self.c2 = EqualizedConv2d(out_ch, out_ch, 3, 1, 1)
            
    def __call__(self, x):
        h = F.unpooling_2d(x, 2, 2, 0, outsize=(x.shape[2]*2, x.shape[3]*2))
        h1 = F.leaky_relu(feature_vector_normalization(self.c0(h)))
        h2 = feature_vector_normalization(self.c1(h1))
        h = F.leaky_relu(h1 + h2) # Resnet
        h = F.leaky_relu(feature_vector_normalization(self.c2(h)))
        return h


# In[6]:


def feature_vector_normalization(x, eps=1e-8):
    # x: (B, C, H, W)
    alpha = 1.0 / F.sqrt(F.mean(x*x, axis=1, keepdims=True) + eps)
    return F.broadcast_to(alpha, x.data.shape) * x


def minibatch_std(x):
    m = F.mean(x, axis=0, keepdims=True)
    div = x - F.broadcast_to(m, x.shape)
    v = F.mean(div*div, axis=0, keepdims=True)
    std = F.mean(F.sqrt(v + 1e-8), keepdims=True)
    std = F.broadcast_to(std, (x.shape[0], 1, x.shape[2], x.shape[3]))
    return F.concat([x, std], axis=1)


# In[7]:


class Generator(chainer.Chain):
    def __init__(self, n_hidden=512, max_stage=12):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.R = (512,512,512,512,256,128,64,32)
        self.max_stage = max_stage
        with self.init_scope():
            
            self.c0 = EqualizedConv2d(n_hidden, self.R[0], 4, 1, 3)
            self.c1 = EqualizedConv2d(self.R[0], self.R[0], 3, 1, 1)
            self.out0 = EqualizedConv2d(self.R[0], 3, 1, 1, 0)

            self.b1 = GeneratorBlock(self.R[0], self.R[1])
            self.out1 = EqualizedConv2d(self.R[1], 3, 1, 1, 0)
            self.b2 = GeneratorBlock(self.R[1], self.R[2])
            self.out2 = EqualizedConv2d(self.R[2], 3, 1, 1, 0)
            self.b3 = GeneratorBlock(self.R[2], self.R[3])
            self.out3 = EqualizedConv2d(self.R[3], 3, 1, 1, 0)
            self.b4 = GeneratorBlock(self.R[3], self.R[4])
            self.out4 = EqualizedConv2d(self.R[4], 3, 1, 1, 0)
            self.b5 = GeneratorBlock(self.R[4], self.R[5])
            self.out5 = EqualizedConv2d(self.R[5], 3, 1, 1, 0)
            self.b6 = GeneratorBlock(self.R[5], self.R[6])
            self.out6 = EqualizedConv2d(self.R[6], 3, 1, 1, 0)
            self.b7 = GeneratorBlock(self.R[6], self.R[7])
            self.out7 = EqualizedConv2d(self.R[7], 3, 1, 1, 0)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.n_hidden, 1, 1))             .astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        return z

    def __call__(self, z, stage):
        # stage0: c0->c1->out0
        # stage1: c0->c1-> (1-a)*(up->out0) + (a)*(b1->out1)
        # stage2: c0->c1->b1->out1
        # stage3: c0->c1->b1-> (1-a)*(up->out1) + (a)*(b2->out2)
        # stage4: c0->c1->b2->out2
        # ...

        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        h = F.reshape(z,(len(z), self.n_hidden, 1, 1))
        h = F.leaky_relu(feature_vector_normalization(self.c0(h)))
        h = F.leaky_relu(feature_vector_normalization(self.c1(h)))

        for i in range(1, int(stage//2+1)):
            h = getattr(self, "b%d"%i)(h)

        if int(stage)%2==0:
            out = getattr(self, "out%d"%(stage//2))
            x = out(h)
        else:
            out_prev = getattr(self, "out%d"%(stage//2))
            out_curr = getattr(self, "out%d"%(stage//2+1))
            b_curr = getattr(self, "b%d"%(stage//2+1))

            x_0 = out_prev(F.unpooling_2d(h, 2, 2, 0, outsize=(2*h.shape[2], 2*h.shape[3])))
            x_1 = out_curr(b_curr(h))
            x = (1.0-alpha)*x_0 + alpha*x_1

        if chainer.configuration.config.train:
            return x
        else:
            scale = int(256 // x.data.shape[2])
            return F.unpooling_2d(x, scale, scale, 0, outsize=(256,256))


# In[8]:


class DiscriminatorBlock(chainer.Chain):
    # conv-conv-downsample
    def __init__(self, in_ch, out_ch, pooling_comp):
        super(DiscriminatorBlock, self).__init__()
        self.pooling_comp = pooling_comp
        self.out_ch = out_ch
        with self.init_scope():
            self.c0 = EqualizedConv2d(in_ch, in_ch, 3, 1, 1)
            self.c1 = EqualizedConv2d(in_ch, in_ch, 3, 1, 1)
            self.c2 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
            
    def __call__(self, x):
        h1 = F.leaky_relu(self.c0(x))
        h2 = self.c1(h1)
        h = F.leaky_relu(h1 + h2) # Resnet
        h = F.leaky_relu(self.c2(h))
        h = self.pooling_comp * F.average_pooling_2d(h, 2, 2, 0)
        return h


# In[9]:


class Discriminator(chainer.Chain):
    def __init__(self, max_stage=12, pooling_comp=1.0):
        super(Discriminator, self).__init__()
        self.max_stage = max_stage
        self.pooling_comp = pooling_comp # compensation of ave_pool is 0.5-Lipshitz
        
        self.R = (32,64,128,256,512,512,512,512)
        
        with self.init_scope():
            self.in7 = EqualizedConv2d(3, self.R[0], 1, 1, 0)
            self.b7 = DiscriminatorBlock(self.R[0], self.R[1], pooling_comp)
            self.in6 = EqualizedConv2d(3, self.R[1], 1, 1, 0)
            self.b6 = DiscriminatorBlock(self.R[1], self.R[2], pooling_comp)
            self.in5 = EqualizedConv2d(3, self.R[2], 1, 1, 0)
            self.b5 = DiscriminatorBlock(self.R[2], self.R[3], pooling_comp)
            self.in4 = EqualizedConv2d(3, self.R[3], 1, 1, 0)
            self.b4 = DiscriminatorBlock(self.R[3], self.R[4], pooling_comp)
            self.in3 = EqualizedConv2d(3, self.R[4], 1, 1, 0)
            self.b3 = DiscriminatorBlock(self.R[4], self.R[5], pooling_comp)
            self.in2 = EqualizedConv2d(3, self.R[5], 1, 1, 0)
            self.b2 = DiscriminatorBlock(self.R[5], self.R[6], pooling_comp)
            self.in1 = EqualizedConv2d(3, self.R[6], 1, 1, 0)
            self.b1 = DiscriminatorBlock(self.R[6], self.R[7], pooling_comp)
            self.in0 = EqualizedConv2d(3, self.R[7], 1, 1, 0)

            self.out0 = EqualizedConv2d(self.R[7]+1, self.R[7], 3, 1, 1)
            self.out1 = EqualizedConv2d(self.R[7], self.R[7], 4, 1, 0)
            self.out2 = EqualizedLinear(self.R[7], 1)

    def __call__(self, x, stage):
        # stage0: in0->m_std->out0_0->out0_1->out0_2
        # stage1: (1-a)*(down->in0) + (a)*(in1->b1) ->m_std->out0->out1->out2
        # stage2: in1->b1->m_std->out0_0->out0_1->out0_2
        # stage3: (1-a)*(down->in1) + (a)*(in2->b2) ->b1->m_std->out0->out1->out2
        # stage4: in2->b2->b1->m_std->out0->out1->out2
        # ...

        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        if int(stage)%2==0:
            fromRGB = getattr(self, "in%d"%(stage//2))
            h = F.leaky_relu(fromRGB(x))
        else:
            fromRGB0 = getattr(self, "in%d"%(stage//2))
            fromRGB1 = getattr(self, "in%d"%(stage//2+1))
            b1 = getattr(self, "b%d"%(stage//2+1))


            h0 = F.leaky_relu(fromRGB0(self.pooling_comp * F.average_pooling_2d(x, 2, 2, 0)))
            h1 = b1(F.leaky_relu(fromRGB1(x)))
            h = (1-alpha)*h0 + alpha*h1

        for i in range(int(stage // 2), 0, -1):
            h = getattr(self, "b%d" % i)(h)

        h = minibatch_std(h)
        h = F.leaky_relu((self.out0(h)))
        h = F.leaky_relu((self.out1(h)))
        return self.out2(h)


# In[10]:


def copy_param(target_link, source_link):
    """Copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] = param.data

    # Copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[:] = link.avg_mean
            target_bn.avg_var[:] = link.avg_var


def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] *= (1 - tau)
        target_params[param_name].data[:] += tau * param.data

    # Soft-copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[:] *= (1 - tau)
            target_bn.avg_mean[:] += tau * link.avg_mean
            target_bn.avg_var[:] *= (1 - tau)
            target_bn.avg_var[:] += tau * link.avg_var


# In[11]:


init=620000

gen = Generator(512)
generator_smooth = Generator(512)
if init != 0:
    serializers.load_npz("generator.npz", gen)
    

gen.compute_accuracy = False
generator_smooth.compute_accuracy = False

gpu = 0
chainer.cuda.get_device(gpu).use()

gen.to_gpu(gpu) # GPUを使うための処理
generator_smooth.to_gpu(gpu) # GPUを使うための処理

# コピー
copy_param(generator_smooth, gen)

if init != 0:
    serializers.load_npz("generator_smooth.npz", generator_smooth)

gen_optimizer = optimizers.Adam(alpha=0.001, beta1=0.0, beta2=0.99)
#gen_optimizer = optimizers.Adam(alpha=0.01, beta1=0.0, beta2=0.99)
gen_optimizer.setup(gen)



# Critic
cri = Discriminator()
cri.compute_accuracy = False
if init != 0:
    serializers.load_npz("critic.npz", cri)

    
cri.to_gpu(gpu) # GPUを使うための処理

cri_optimizer = optimizers.Adam(alpha=0.001, beta1=0.0, beta2=0.99) # 0.001
#cri_optimizer = optimizers.Adam(alpha=0.01, beta1=0.0, beta2=0.99)

cri_optimizer.setup(cri)
#cri_optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001)) # 0.005




# In[12]:


mloader = MultiLoader(passA='./train',passB='')
total_num = mloader.num
mloader.fileListA


# In[13]:


def real_image(images, stage):


    if math.floor(stage)%2==0:
        reso = min(256, 4 * 2**(((math.floor(stage)+1)//2)))
        scale = max(1, 256//reso)
        if scale>1:
            images = F.average_pooling_2d(images, scale, scale, 0)
    else:
        alpha = stage - math.floor(stage)
        reso_low = min(256, 4 * 2**(((math.floor(stage))//2)))
        reso_high = min(256, 4 * 2**(((math.floor(stage)+1)//2)))
        scale_low = max(1, 256//reso_low)
        scale_high = max(1, 256//reso_high)
        
        images_low = F.unpooling_2d(F.average_pooling_2d(x_real, scale_low, scale_low, 0), 2, 2, 0, outsize=(reso_high, reso_high))
        
        if scale_high > 1:
            images_high = F.average_pooling_2d(images, scale_high, scale_high, 0)
        else:
            images_high = images
        images = (1-alpha)*images_low + alpha*images_high
        
    return images


# In[14]:


print(total_num)


# In[15]:


# シード値設定
np.random.seed(seed=20180101)
xp.random.seed(seed=20180101)
print(sum(p.data.size for p in gen.params()))
print(sum(p.data.size for p in cri.params()))


# In[ ]:


def getCount2(init, batches, interval):
    counter = 0
    hoge = init
    for i in range(len(batches)*2):
        tmp = interval / batches[(i+1)//2]

        hoge = hoge - tmp
        
        if hoge > 0:
            counter += interval
        else:
            hoge = hoge + tmp
            counter += hoge*batches[(i+1)//2]
            break
    return (int)(counter)


# In[ ]:


batch_size = 16
batches = (16,16,16,16,16,9,5)
n_critic = 1

g_loss = 0
c_loss = 0
w_dis = 0
w_dis_sum = 0

grad_loss = 0

smoothing = 0.999

zTest = generator_smooth.make_hidden(16)
zTest = Variable(zTest)

stage=0
if os.path.isfile('stage.txt') and init!=0:
    stage = np.loadtxt('stage.txt')
    
test_interval = 100

save_interval = 5000

# w_dis保存用
w_dis_list = np.empty((0,2))
if os.path.isfile('log.csv') and init!=0:
    w_dis_list = np.loadtxt('log.csv', delimiter=',')


# 時間計測
A = time.time()


# stage学習枚数
interval = 800000
counter = getCount2(init, batches, interval)

for i in range(10000000):
    
    stage = counter / interval
    #stage=11.5
    #stage = (init+i) / 100000.0
    
    if stage >= 13:
        stage = 12
        
    #stage=11.5
    
    batch_size = batches[int((stage+1)//2)]
    #stage = 11.5
    # critic更新   
    
    # 本物用
    x_real = mloader.getBatch(batch_size=batch_size, alpha=1.0)
    #x_real = cvBatch.nextBatch(batch_size)
    x_real = Variable(xp.asarray(x_real, dtype=np.float32))

    # スケール合わせ
    x_real = real_image(x_real, stage)
    y_real = cri(x_real, stage=stage)
    
    z = gen.make_hidden(batch_size)
    z = Variable(z)

    # 偽物用
    x_fake = gen(z, stage=stage)
    y_fake = cri(x_fake, stage=stage)

    # BackPropの参照を削除
    x_fake.unchain_backward()

    # grad
    eps = xp.random.uniform(0, 1, size=batch_size).astype("f")[:, None, None, None]
    x_mid = eps * x_real + (1.0 - eps) * x_fake

    x_mid_v = Variable(x_mid.data)
    y_mid = F.sum(cri(x_mid_v, stage=stage))

    dydx, = chainer.grad([y_mid], [x_mid_v], enable_double_backprop=True)
    dydx = F.sqrt(F.sum(dydx*dydx, axis=(1, 2, 3)))
    #loss_grad = 10.0 * F.mean_squared_error(dydx, 750.0 * xp.ones_like(dydx.data)) * (1.0/750.0**2)
    loss_grad = 10.0 * F.mean_squared_error(dydx, xp.ones_like(dydx.data))

    # 0.001
    loss_dr = 0.001 * F.sum(y_real**2) / batch_size

    real_loss = F.sum(-y_real) / batch_size
    fake_loss = F.sum(y_fake) / batch_size
    L_cri = real_loss
    L_cri += fake_loss
    
    # w距離
    w = -fake_loss.data - real_loss.data
    w_dis += w
    w_dis_sum += w

    L_cri += loss_dr

    c_loss += (L_cri.data)

    L_cri +=loss_grad

    grad_loss+=loss_grad.data
    
    

    cri.cleargrads()
    L_cri.backward()
    cri_optimizer.update()

    # BackPropの参照を削除
    L_cri.unchain_backward()
    
    # 偽物用
    z = gen.make_hidden(batch_size)
    z = Variable(z)
        
    x_fake = gen(z, stage=stage)
    y_fake = cri(x_fake, stage=stage)

    L_gen = F.sum(-y_fake) / batch_size
    
    g_loss += (L_gen.data)
    
    gen.cleargrads()
    L_gen.backward()
    gen_optimizer.update()
    
    # BackPropの参照を削除
    L_gen.unchain_backward()
    
    # update smoothed generator
    soft_copy_param(generator_smooth, gen, 1.0-smoothing)
    
    if (i+init) % save_interval == 0 and i != 0:
        print("save")
        gen.to_cpu()
        serializers.save_npz("generator.npz", gen)
        gen.to_gpu(gpu)
        
        generator_smooth.to_cpu()
        serializers.save_npz("generator_smooth.npz", generator_smooth)
        generator_smooth.to_gpu(gpu)
        
        cri.to_cpu()
        serializers.save_npz("critic.npz", cri)
        cri.to_gpu(gpu)
        # 現在のステージを保存
        np.savetxt('stage.txt', np.array([stage]))
        print("ok")
        
    if i % 500 == 0 and i != 0:
        # w_dis保存
        w_dis_list = np.append(w_dis_list, np.array([[int(init+i), w_dis_sum/500]]), axis=0)
        np.savetxt('log.csv', w_dis_list,delimiter=',')
        w_dis_sum = 0
    
    if (init+i) % 20000 == 0 and i != 0:
        print("save")
        gen.to_cpu()
        serializers.save_npz("./weight/generator_" + str(init+i) + ".npz", gen)
        gen.to_gpu(gpu)
        
        generator_smooth.to_cpu()
        serializers.save_npz("./weight/generator_smooth_" + str(init+i) + ".npz", generator_smooth)
        generator_smooth.to_gpu(gpu)
        
        cri.to_cpu()
        serializers.save_npz("./weight/critic_" + str(init+i) + ".npz", cri)
        cri.to_gpu(gpu)
        # 現在のステージを保存
        np.savetxt('stage.txt', np.array([stage]))
        print("ok")
    
    if i % test_interval == 0:

        # 画像出力
        # テストモードに
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            
            # 偽物用
            x_test = generator_smooth(zTest, stage)
            #y_test = cri(x_test)
            
            size = 4*2**((int(stage)+1)//2)
            
            print(str(init+i) + ",epoch:" + str(int(batch_size/total_num*(init+i))) + ",gen:" + str(g_loss/test_interval) + 
                  ", cri:" + str(c_loss/test_interval) + ", w_dis:" + str(w_dis/test_interval) + ", grad_loss:" + str(grad_loss/test_interval))
            print("stage:" + str(stage) + ", size:" + str(size) + ", counter:" + str(counter))
            
            g_loss = 0
            c_loss = 0
            w_dis = 0
            grad_loss = 0
            
        imgs = cuda.to_cpu(x_test.data)
        imgs = np.transpose(imgs, (0, 3, 2, 1))

        # 画像を連結
        img2 = []
        
        s = 4

        for k in range(s):
            img2.append(imgs[k*s])

            for j in range(s):
                if j != 0:
                    img2[k] = cv2.hconcat([img2[k], imgs[k*s+j]])
            if k == 0:
                img3 = img2[0]
            else:
                img3 = cv2.vconcat([img3, img2[k]])

        img3 = np.asarray(np.clip(img3 * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        cv2.imwrite('./res/'+str(init+i)+'.jpg', img3)
        
        # かかった時間
        print("time:{0} sec".format(time.time()-A))
        
        A = time.time()
        
    counter+=batch_size 


# In[ ]:


gen.to_cpu()
serializers.save_npz("generator.npz", gen)
cri.to_cpu()
serializers.save_npz("critic.npz", cri)

