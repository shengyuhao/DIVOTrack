import os

results_path = './'
save_path = './resize/'
files = os.listdir(results_path)
for file in files:
    f = open(results_path + file, 'r')
    datas = [i.split(',') for i in f.readlines()]
    save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
    lines = []

    for data in datas:
        fid = data[0]
        pid = data[1]
        x = float(data[2])
        y = float(data[3])
        w = float(data[4]) - x
        h = float(data[5]) - y
        if 'View2' in file:
            lines.append(save_format.format(frame=fid, id=pid, x1=(x*1920)/3640, y1=(y*1080)/2048, w=(w*1920)/3640, h=(h*1080)/2048))
        else:
            lines.append(save_format.format(frame=fid, id=pid, x1=x, y1=y, w=w, h=h))
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
    with open(os.path.join(save_path, file), 'w') as f:
        f.writelines(lines)
        