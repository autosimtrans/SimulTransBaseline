from IPython import embed


def delay(act):
    # compute ap, cw and al for prediction
    x, y = float(act.count(0)), float(act.count(1))
    #print ('read ', x)
    #print ('write ', y)
    #print (pred)
    #print ('------------------------------------------')
    if x == 0: x = 0.000001
    if y == 0: y = 0.000001
    g_t = []
    c, seg = 0, 0
    last0, last1 = 0, 0
    for j, a in enumerate(act):
        if a == 0:
            c += 1
            last0 = j
            if j + 1 < len(act) and act[j+1] == 1:
                seg += 1
        else:
            g_t.append(c)
            last1 = j
    ap = sum(g_t) / (x * y)
    if seg == 0: seg = 0.0000001
    cw = x / seg
    r = y / x
    tail = last0 + 2 if last1 > last0 else last1 + 1
    tau = act[:tail].count(1)
    g_t = g_t[:tau]
    tau = float(tau) if tau != 0 else 0.0000001
    temp = [g - (t / r) for t, g in enumerate(g_t)]
    al = sum(temp) / tau
    return ap, cw, al

#print ('en2de')
def baigong():

    directs = ['z2e', 'e2z', 'd2e', 'e2d']
    sets = ['dev', 'test']
    beam = [1, 5]
    for b in beam:
        for d in directs:
            if 'd' in d:
                ks = range(1, 11)
            else:
                ks = range(1, 11)
                #ks = range(1, 10, 2)

            for s in sets:
                print (d, s, 'beam size =', b)
                wk_al, wk_ap, wk_cw = [], [], []
                if d == 'z2e' and s == 'dev':
                    file1 = '/mnt/home/baigong/scratch_SMT/data/zh2en/1M/orig_bpe/dev_06.zh.bpe'
                    file22 = '/mnt/home/baigong/scratch_SMT/sl_policy_transformer/dec_zh2en/dev_w'
                    #file1 = '/mnt/data/mam/data/NIST_zh2en_2M/testdata/bped/test_06.zh.bpe'
                    #file22 = '/mnt/home/zrenj/Project/tail-beam/dec_zh2en/dev_w'
                elif d == 'z2e' and s == 'test':
                    file1 = '/mnt/home/baigong/scratch_SMT/data/zh2en/1M/orig_bpe/test.zh.bpe'
                    file22 = '/mnt/home/baigong/scratch_SMT/sl_policy_transformer/dec_zh2en/test_w'
                    #file1 = '/mnt/data/mam/data/NIST_zh2en_2M/testdata/bped/test_08.zh.bpe'
                    #file22 = '/mnt/home/zrenj/Project/tail-beam/dec_zh2en/test_w'
                elif d == 'e2z' and s == 'dev':
                    file1 = '/mnt/home/baigong/scratch_SMT/data/zh2en/1M/orig_bpe/dev_tgt_1.bpe'
                    file22 = '/mnt/home/baigong/scratch_SMT/sl_policy_transformer/dec_en2zh/dev_w'
                    #file1 = '/mnt/data/mam/data/NIST_zh2en_2M/testdata/bped/test_06.en.1.bpe'
                    #file22 = '/mnt/home/zrenj/Project/tail-beam/dec_en2zh/dev_w'
                elif d == 'e2z' and s == 'test':
                    file1 = '/mnt/home/baigong/scratch_SMT/data/zh2en/1M/orig_bpe/test_tgt_1.bpe'
                    file22 = '/mnt/home/baigong/scratch_SMT/sl_policy_transformer/dec_en2zh/test_w'
                    #file1 = '/mnt/data/mam/data/NIST_zh2en_2M/testdata/bped/test_08.en.1.bpe'
                    #file22 = '/mnt/home/zrenj/Project/tail-beam/dec_en2zh/test_w'
                elif d == 'd2e' and s == 'dev':
                    file1 = '/mnt/data/mam/data/wmt15en2de/wmt13/wmt13.test.de.txt.bpe'
                    file22 = '/mnt/home/baigong/scratch_SMT/sl_policy_transformer/dec_de2en/dev_w'
                    #file1 = '/mnt/data/mam/data/wmt15en2de/wmt13/wmt13.test.de.txt.bpe'
                    #file22 = '/mnt/home/zrenj/Project/tail-beam/dec_de2en/dev_w'
                elif d == 'd2e' and s == 'test':
                    file1 = '/mnt/home/baigong/scratch_SMT/data/wmt15de-en/org/en2de.test.de.bpe.txt'
                    file22 = '/mnt/home/baigong/scratch_SMT/sl_policy_transformer/dec_de2en/test_w'
                    #file1 = '/mnt/data/mam/data/wmt15en2de/en2de.test.de.bpe.txt'
                    #file22 = '/mnt/home/zrenj/Project/tail-beam/dec_de2en/test_w'
                elif d == 'e2d' and s == 'dev':
                    file1 = '/mnt/data/mam/data/wmt15en2de/wmt13/wmt13.test.en.txt.bpe'
                    file22 = '/mnt/home/baigong/scratch_SMT/sl_policy_transformer/dec_en2de/dev_w'
                    #file1 = '/mnt/data/mam/data/wmt15en2de/wmt13/wmt13.test.en.txt.bpe'
                    #file22 = '/mnt/home/zrenj/Project/tail-beam/dec_en2de/dev_w'
                elif d == 'e2d' and s == 'test':
                    file1 = '/mnt/home/baigong/scratch_SMT/data/wmt15de-en/org/en2de.test.en.bpe.txt'
                    file22 = '/mnt/home/baigong/scratch_SMT/sl_policy_transformer/dec_en2de/test_w'
                    #file1 = '/mnt/data/mam/data/wmt15en2de/en2de.test.en.bpe.txt'
                    #file22 = '/mnt/home/zrenj/Project/tail-beam/dec_en2de/test_w'


                for i in ks:
                    k = i
                    file2 = file22+str(k)+'_b'+str(b)+'.bpe'
                    all_ap, all_cw, all_al = [], [], []
                    with open(file1, 'r') as src, open(file2, 'r') as tgt:
                        for s, t in zip(src, tgt):
                            ss, tt = s.split(), t.split()
                            act = [0] * (k-1)
                            s_idx = k-1
                            for j, x in enumerate(tt):
                                if s_idx < len(ss):
                                    act.append(0)
                                    s_idx += 1
                                act.append(1)
                                if d == 'e2z' and j % 4 == 0 and s_idx < len(ss):
                                    act.append(0)
                                    s_idx += 1
                                if d == 'd2e' and j % 5 == 0 and s_idx < len(ss):
                                    act.append(0)
                                    s_idx += 1
                            if s_idx < len(ss):
                                act += [0] * (len(ss) -s_idx)

                            ap, cw, al = delay(act)

                            all_ap.append(ap)
                            all_cw.append(cw)
                            all_al.append(al)
                        wk_ap.append(sum(all_ap) / len(all_ap))
                        wk_cw.append(sum(all_cw) / len(all_cw))
                        wk_al.append(sum(all_al) / len(all_al))

                print('AP = [', ', '.join(f'{x:.5f}' for x in wk_ap), ']')
                print('AL = [', ', '.join(f'{x:.5f}' for x in wk_al), ']')
                print('CW = [', ', '.join(f'{x:.5f}' for x in wk_cw), ']')
    #print ('ap = ' % wk_ap)
    #print ('al = ' % wk_al)
    #print ('cw = ' % wk_cw)

def latency(file1, file2, k, skip):
    all_ap, all_cw, all_al = [], [], []
    with open(file1, 'r') as src, open(file2, 'r') as tgt:
        for _i, (s, t) in enumerate(zip(src, tgt)):
            ss, tt = s.split(), t.split()
            act = [0] * (k-1)
            s_idx = k-1
            for j, x in enumerate(tt):
                if s_idx < len(ss):
                    act.append(0)
                    s_idx += 1
                act.append(1)
                if skip == 3 and j % 2 == 0 and s_idx < len(ss):
                    # 1 2
                    act.append(0)
                    s_idx += 1
                elif skip == 4 and j % 3 == 0 and s_idx < len(ss):
                    # 1 2 1
                    act.append(0)
                    s_idx += 1
                elif skip == 5 and j % 4 == 0 and s_idx < len(ss):
                    # 1 2 1 1
                    act.append(0)
                    s_idx += 1
                elif skip == 6 and j % 5 == 0 and s_idx < len(ss):
                    # 1 2 1 1 1
                    act.append(0)
                    s_idx += 1
            if s_idx < len(ss):
                act += [0] * (len(ss) -s_idx)

            ap, cw, al = delay(act)
            # print(_i, sum(act), len(act) - sum(act), al, act)

            all_ap.append(ap)
            all_cw.append(cw)
            all_al.append(al)
        ret_ap = sum(all_ap) / len(all_ap)
        ret_cw = sum(all_cw) / len(all_cw)
        ret_al = sum(all_al) / len(all_al)

    return ret_ap, ret_cw, ret_al

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 5:
        f_src = sys.argv[1]
        f_tgt = sys.argv[2]
        k = int(sys.argv[3])
        skip = int(sys.argv[4])
        print('AP: %.4f CW: %.4f AL: %.4f' % latency(f_src, f_tgt, k, skip))

    else:

        f_src = '/mnt/data/mam/data/wmt15en2de/en2de.test.en.bpe.txt'
        f_tgt = 'dec_en2de/test_w1_b1.bpe'
        f_tgt = 'dec_en2de/test_w3_b1.bpe'


        print(latency(f_src, f_tgt, 1, 6))

