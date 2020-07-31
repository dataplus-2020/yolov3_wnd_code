import os
if __name__ == '__main__':
    seeds = [3, 5, 9]
    display_type = 'syn_background'
    syn_ratio = [0.3, 0.2, 0.1, 0]
    for seed in seeds:
        for sr in syn_ratio:
            if seed==5 and t==0 and sr==0.3:
                continue
            if seed==3 and t==0 and sr==0.3:
                continue
            # os.system('python train_syn_background_seeds.py --t %d --seed %d --syn_display_type %s --syn_ratio %.2f' % (t, seed, display_type, sr))
            f = os.popen('python train_syn_background_seeds.py --t %d --seed %d --syn_display_type %s --syn_ratio %.2f' % (t, seed, display_type, sr), 'r')
            d = f.read()
            print(d)
            f.close()
            # os.system('python train_syn_background_seeds.py --t %d --seed %d --syn_display_type %s --syn_ratio %f'
            #           % (t, seed, display_type, sr))
