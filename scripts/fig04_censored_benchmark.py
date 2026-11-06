"""Figure 4: benchmark comparison across censored datasets."""
from figkit_compact import build
from benchmark_panels import panel_a,panel_b,panel_c,panel_d

def layout(fig):
    g=fig.add_gridspec(3,2,width_ratios=[1.05,1],height_ratios=[1,1,1],left=.025,right=.98,bottom=.035,top=.975,hspace=.08,wspace=.08)
    return {'a':g[:,0],'b':g[0,1],'c':g[1,1],'d':g[2,1]}

if __name__=='__main__':
    build([('a',panel_a),('b',panel_b),('c',panel_c),('d',panel_d)],layout,'Fig04_censored_benchmark','Fig04_censored_benchmark',(13.4,11.6),sub_sizes={'a':(8,11.5),'b':(6.4,4.4),'c':(6.4,4.4),'d':(6.4,4.8)})
