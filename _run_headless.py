import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Save every open figure instead of blocking on an interactive window
def _save_all(*a, **k):
    for i in plt.get_fignums():
        fig = plt.figure(i)
        label = fig.get_label() or f'fig{i}'
        fig.savefig(f'run_{i}_{label}.png', dpi=120, bbox_inches='tight')
        print(f'[saved] run_{i}_{label}.png')
plt.show = _save_all

with open('hh_annie_model_june_1_final.py', encoding='utf-8') as f:
    code = f.read()
exec(compile(code, 'hh_annie_model_june_1_final.py', 'exec'), {'__name__': '__main__'})
