"""Figure 3: learning routes and fitted uncertainty mechanism."""
from figkit_compact import build
from learning_panels import panel_a,panel_b,panel_c,panel_d

def layout(fig):
    g=fig.add_gridspec(2,3,height_ratios=[1.0,1.0],width_ratios=[1,1.35,1],left=.025,right=.98,bottom=.035,top=.975,hspace=.08,wspace=.07)
    return {'a':g[0,:],'b':g[1,0],'c':g[1,1],'d':g[1,2]}

if __name__=='__main__':
    build([('a',panel_a),('b',panel_b),('c',panel_c),('d',panel_d)],layout,'Fig03_learning_framework','Fig03_learning_framework',(13.2,9.2),sub_sizes={'a':(17.5,6),'b':(6,4.8),'c':(6.4,5.4),'d':(6,4.8)})
