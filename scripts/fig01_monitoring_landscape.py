"""Figure 1: monitoring coverage and event categories."""
from figkit_compact import build
from monitoring_panels import panel_a,panel_b

def layout(fig):
    g=fig.add_gridspec(2,1,height_ratios=[1.20,0.88],left=.025,right=.98,bottom=.035,top=.975,hspace=.08)
    return {'a':g[0,0],'b':g[1,0]}

if __name__=='__main__':
    build([('a',panel_a),('b',panel_b)],layout,'Fig01_monitoring_landscape','Fig01_monitoring_landscape',(13.2,9.0),sub_sizes={'a':(13.2,5.0),'b':(8.2,5.7)})
