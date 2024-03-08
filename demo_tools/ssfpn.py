        for i in range(4):  # self.fpn_lvl
            att = F.interpolate(super_atts[i], size=(1080,1920), mode='nearest')
            att = (att.detach().cpu().numpy()[0])
            # att /= np.max(att)
            #att = np.power(att, 0.8)
            att = att * 255
            att = att.astype(np.uint8).transpose(1, 2, 0)
            # att = cv2.applyColorMap(att, cv2.COLORMAP_JET)    
            image = imageio.imsave('results/sup{}.jpg'.format(i), att)