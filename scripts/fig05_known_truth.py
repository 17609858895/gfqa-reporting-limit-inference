"""Figure 5: validation against known concentrations."""
from figkit_compact import build
from truth_panels import panel_a,panel_b,panel_c

def layout(fig):
    g=fig.add_gridspec(2,2,height_ratios=[1,.88],left=.025,right=.98,bottom=.035,top=.975,hspace=.08,wspace=.06)
    return {'a':g[0,0],'b':g[0,1],'c':g[1,:]}

if __name__=='__main__':
    build([('a',panel_a),('b',panel_b),('c',panel_c)],layout,'Fig05_known_truth','Fig05_known_truth',(12.8,9.5),sub_sizes={'a':(6,4.8),'b':(6,4.8),'c':(7.8,5.2)})
