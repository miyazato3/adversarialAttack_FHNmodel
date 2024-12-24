import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os

# パラメータ設定
network_list = ["WSp=0.txt", "WSp=0.006.txt", "WSp=0.232.txt", "WSp=1.txt", "FRACTAL.txt"]
attack_eps = [0.0, 0.05, -0.05]
tmax = 10000
dt = 0.5
t_interval = 1.0
seed = 128

os.makedirs("results", exist_ok=True)

# シミュレーション実行
for network in network_list:
    for e in attack_eps:
        out = subprocess.run(
            [
                "python", "advFHNOscillator.py",
                *[
                    "--network", f"{network}",
                    "--N", f"{90 if network != 'FRACTAL.txt' else 82}",
                    "--sigma", f"{0.0506 if network != 'FRACTAL.txt' else 0.01}",
                    "--epsilon", "0.05",
                    "--a", "0.5",
                    "--tmax", f"{tmax}",
                    "--dt", f"{dt}",
                    "--t_interval", f"{t_interval}",
                    "--attack_eps", f"{e}",
                    "--seed", f"{seed}"
                ]
            ],
            text=True, capture_output=True, check=True
        )
        print(out)

    # 結果を読み込む
    r_values_base = np.loadtxt(f"results/r_values_{network[:-4]}_eps={attack_eps[0]}_seed={seed}.txt")
    t_values_base = np.loadtxt(f"results/t_values_{network[:-4]}_eps={attack_eps[0]}_seed={seed}.txt")
    r_values_pos = np.loadtxt(f"results/r_values_{network[:-4]}_eps={attack_eps[1]}_seed={seed}.txt")
    t_values_pos = np.loadtxt(f"results/t_values_{network[:-4]}_eps={attack_eps[1]}_seed={seed}.txt")
    r_values_neg = np.loadtxt(f"results/r_values_{network[:-4]}_eps={attack_eps[2]}_seed={seed}.txt")
    t_values_neg = np.loadtxt(f"results/t_values_{network[:-4]}_eps={attack_eps[2]}_seed={seed}.txt")
    
    # 結果をプロット
    plt.figure(figsize=(8, 4))
    plt.plot(t_values_base, r_values_base, alpha=0.6, linewidth=0.1, label="$\epsilon=0$")
    plt.plot(t_values_pos, r_values_pos, alpha=0.6, linewidth=0.1, label="$\epsilon=0.05$")
    plt.plot(t_values_neg, r_values_neg, alpha=0.6, linewidth=0.1, label="$\epsilon=-0.05$")
    plt.xlabel("Time $t$")
    plt.ylabel("Order Parameter $r$")
    plt.ylim(0,1)
    legend = plt.legend(loc="lower left", bbox_to_anchor=(1.05, 0), borderaxespad=0.)
    for legend_line in legend.get_lines():
        legend_line.set_linewidth(2.0)
        legend_line.set_alpha(1.0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"results/{network[:-4]}_tmax{tmax}_dt{dt}_tinterval{t_interval}_seed{seed}.png", dpi=500)
