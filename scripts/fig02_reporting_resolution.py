"""Figure 2: reporting resolution and interval categories."""
from figkit_compact import build
from monitoring_panels import panel_c,panel_d,panel_e

def layout(fig):
    g=fig.add_gridspec(2,2,height_ratios=[1.0,.52],left=.025,right=.98,bottom=.035,top=.965,hspace=.08,wspace=.06)
    return {'a':g[0,0],'b':g[0,1],'c':g[1,:]}

if __name__=='__main__':
    build([('a',panel_c),('b',panel_d),('c',panel_e)],layout,'Fig02_reporting_resolution','Fig02_reporting_resolution',(13.2,7.7),sub_sizes={'a':(6.2,5.0),'b':(6.2,5.0),'c':(13.2,3.5)})
