import numpy as np
import subprocess
import matplotlib.pyplot as plt
import datetime
import os

# パラメータ設定
#network_list = ["WSp=0.txt", "WSp=0.006.txt", "WSp=0.232.txt", "WSp=1.txt", "FRACTAL.txt"]
network_list = ["WSp=0.006.txt"]
attack_eps = [0.0, 0.05, -0.05]
tmax = 10000
dt = 0.5
#t_interval = 1.0
t_interval = 100.0
seed = 128
random = True

# 結果の保存先を設定
root_dir = "results"
current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_path = f"{root_dir}/{current_time}"
os.makedirs(save_path, exist_ok=True)

# シミュレーション実行
for i, network in enumerate(network_list):
    for j, e in enumerate(attack_eps):
        args = [
            "--network", f"{network}",
            "--N", f"{90 if network != 'FRACTAL.txt' else 82}",
            "--sigma", f"{0.0506 if network != 'FRACTAL.txt' else 0.01}",
            "--epsilon", "0.05",
            "--a", "0.5",
            "--tmax", f"{tmax}",
            "--dt", f"{dt}",
            "--t_interval", f"{t_interval}",
            "--attack_eps", f"{e}",
            "--seed", f"{seed}",
            "--save_path", f"{save_path}",
        ]
        if random: args.append("--random")
        if i==0 and j==0: args.append("--export_t")

        out = subprocess.run(
            ["python", "advFHNOscillator.py", *args],
            text=True, capture_output=True, check=True
        )
        print(out)

    # 結果を読み込む
    r_values_base = np.loadtxt(f"{save_path}/r_values_{network[:-4]}_eps={attack_eps[0]}_seed={seed}_random={random}.txt")
    r_values_pos = np.loadtxt(f"{save_path}/r_values_{network[:-4]}_eps={attack_eps[1]}_seed={seed}_random={random}.txt")
    r_values_neg = np.loadtxt(f"{save_path}/r_values_{network[:-4]}_eps={attack_eps[2]}_seed={seed}_random={random}.txt")
    t_values = np.arange(0, tmax, dt)
    
    # 結果をプロット
    plt.figure(figsize=(8, 4))
    plt.plot(t_values, r_values_base, alpha=0.6, linewidth=0.1, label="$\epsilon=0$")
    plt.plot(t_values, r_values_pos, alpha=0.6, linewidth=0.1, label="$\epsilon=0.05$")
    plt.plot(t_values, r_values_neg, alpha=0.6, linewidth=0.1, label="$\epsilon=-0.05$")
    plt.title(f"{network[:-4]}")
    plt.xlabel("Time $t$")
    plt.ylabel("Order Parameter $r$")
    plt.ylim(0,1)
    legend = plt.legend(loc="lower left", bbox_to_anchor=(1.05, 0), borderaxespad=0.)
    for legend_line in legend.get_lines():
        legend_line.set_linewidth(2.0)
        legend_line.set_alpha(1.0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{save_path}/{network[:-4]}_tmax{tmax}_dt{dt}_tinterval{t_interval}_seed{seed}.png", dpi=500)
