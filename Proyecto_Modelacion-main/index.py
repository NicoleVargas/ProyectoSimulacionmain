

from flask import Flask, render_template, request, send_file

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def home():
    return render_template('home.html')
    

@app.route('/cuadradosMedios')
def cuadradosMedios():
    return render_template('cuadradosMedios.html')

@app.route('/MED')
def MED():
    return render_template('MED.html')


@app.route('/imprimirCuadradosMedios')
def imprimirCuadradosMedios():
    return render_template('imprimirCuadradosMedios.html')

@app.route('/calcularCuadradosMedios', methods=['GET','POST'])
def calcularCuadradosMedios():
    n = request.form.get('numeroIteraciones', type=int)
    r = request.form.get('semilla', type=int)

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO 
    import base64

    # n=100
    # r=23456 
    l=len(str(r))
    lista = []
    lista2 = []
    i=1
    while i <= n:
        x=str(r*r)
        if l % 2 == 0:
            x = x.zfill(l*2)
        else:
            x = x.zfill(l)
        y=(len(x)-l)/2
        y=int(y)
        r=int(x[y:y+l])
        lista.append(r)
        lista2.append(x)
        i=i+1  
    df = pd.DataFrame({'Valores elevados':lista2,'Valor medio':lista})
    dfrac = df["Valor medio"]/10**l
    df['Valor random'] = dfrac

    buf = io.BytesIO()
    x1=df['Valor random']
    plt.plot(x1)
    plt.title('Generador de Números Aleatorios Cuadrados Medios')
    plt.xlabel('Serie')
    plt.ylabel('Aleatorios')
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data= df.to_html(classes="table table-light table-striped", justify="justify-all", border=0)

    writer = ExcelWriter("static/file/data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()
            
    df.to_csv("static/file/data.csv", index=False) 

    return render_template('imprimirCuadradosMedios.html', data=data, image=plot_url)


@app.route('/congruencialLineal')
def congruencialLineal():
    return render_template('congruencialLineal.html')

@app.route('/imprimirCongruencialLineal')
def imprimirCongruencialLineal():
    return render_template('imprimirCongruencialLineal.html')

@app.route('/calcularCongruencialLineal', methods=['GET','POST'])
def calcularCongruencialLineal():
    n = request.form.get("numeroIteraciones", type=int)
    x0 = request.form.get("semilla", type=int) 
    a = request.form.get("multiplicador", type=int)
    c = request.form.get("incremento", type=int)
    m = request.form.get("modulo", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO 
    import base64

    #n, m, a, x0, c = 20,1000,101,4,457
    x = [1]*n
    r = [0.1]*n
    for i in range(0,n):
            x[i] = ((a*x0)+c) % m
            x0 = x[i]
            r[i] = x0/m
    df = pd.DataFrame({'Xn': x, 'ri':r})
    
    # Graficamos los numeros generados
    buf = io.BytesIO()
    plt.plot(r,marker='o')
    plt.title('Generador de Números Aleatorios Congruencial Lineal')
    plt.xlabel('Serie')
    plt.ylabel('Aleatorios')
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data= df.to_html(classes="table table-light table-striped", justify="justify-all", border=0)

    writer = ExcelWriter("static/file/data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()
            
    df.to_csv("static/file/data.csv", index=False) 

    return render_template('imprimirCongruencialLineal.html', data=data, image=plot_url)


@app.route('/congruencialMultiplicativo')
def congruencialMultiplicativo():
    return render_template('congruencialMultiplicativo.html')

@app.route('/imprimirCongruencialMultiplicativo')
def imprimirCongruencialMultiplicativo():
    return render_template('imprimirCongruencialMultiplicativo.html')

@app.route('/calcularCongruencialMultiplicativo', methods=['GET','POST'])
def calcularCongruencialMultiplicativo():
    n = request.form.get("numeroIteraciones", type=int)
    x0 = request.form.get("semilla", type=int) 
    a = request.form.get("multiplicador", type=int)
    m = request.form.get("modulo", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO 
    import base64

    # n, m, a, x0 = 20, 1000, 747, 123
    x = [1] * n
    r = [0.1] * n
    for i in range(0, n):
     x[i] = (a*x0) % m
     x0 = x[i]
     r[i] = x0 / m
    d = {'Xn': x, 'ri': r }
    df = pd.DataFrame(data=d)

    buf = io.BytesIO()
    plt.plot(r,'g-', marker='o',)
    plt.title('Generador de Números Aleatorios Congruencial Multiplicativo')
    plt.xlabel('Serie')
    plt.ylabel('Aleatorios')
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data= df.to_html(classes="table table-light table-striped", justify="justify-all", border=0)
    
    writer = ExcelWriter("static/file/data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()          
    
    df.to_csv("static/file/data.csv", index=False) 

    return render_template('imprimirCongruencialMultiplicativo.html', data=data, image=plot_url)


######## Promedisos Inicio   ####
@app.route('/prommovil', methods=("POST", "GET"))
def prommovil():

    from flask import Blueprint, Flask, render_template, make_response, request, send_file
    from wtforms import Form, FloatField, validators,StringField, IntegerField
    from numpy import exp, cos, linspace
    from math import pi
    import io
    import random
    import os, time, glob
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class InputForm(Form):
        N = StringField(
            label='Escriba los valores a ingresar separados por comas (,)', default='5, 6 ,8, 7, 9, 10',
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        ### Promedio Móvil
        # Vamos a crear un DataFrame con los datos y luego procederemos a calcular el promedio movil MMO_3 = 3 y MMO_4 = 4
        # el DataFrame se llama movil
        prueba = str(form.N.data)
        ex = prueba.split(",")
        ex2 =list(map(float,ex))
        exporta = {'Año':ex2,
        'Valores':ex2}
        movil = pd.DataFrame(exporta)

        prediccion = len(ex2)
        predfinal = prediccion - 3
        # mostramos los 5 primeros registros
        # calculamos para la primera media móvil MMO_3
        for i in range(0,movil.shape[0]-2):
            movil.loc[movil.index[i+2],'Promedio a 3'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1])/3),1)
        # calculamos para la segunda media móvil MMO_4
        for i in range(0,movil.shape[0]-3):
            movil.loc[movil.index[i+3],'Promedio a 4'] = np.round(((movil.iloc[i,1]+movil.iloc[i+1,1]+movil.iloc[i+2,1]+movil.iloc[i+
        3,1])/4),1)
        # calculamos la proyeción final
        proyeccion = movil.iloc[predfinal:,[1,2,3]]
        p1,p2,p3 =proyeccion.mean()
        # incorporamos al DataFrame
        a = movil.append({'Año':2018,'Valores':p1, 'Promedio a 3':p2, 'Promedio a 4':p3},ignore_index=True)
        # mostramos los resultados
        a['Error promedio 3'] = a['Valores']-a['Promedio a 3']
        a['Error promedio 4'] = a['Valores']-a['Promedio a 4']
        df = a

        dftemp1 = df['Valores']
        dftemp2 = df['Promedio a 3']
        dftemp3 = df['Promedio a 4']
        dftemp4 = df['Error promedio 3']
        dftemp5 = df['Error promedio 4']
        resv1 = dftemp1[0]
        resv2 = dftemp1[1]
        resv3 = dftemp1[2]
        resv4 = dftemp1[3]

        resv5 = dftemp2[2]
        resv6 = dftemp3[3]
        resv7 = dftemp4[2]
        resv8 = dftemp5[3]

        plt.figure(figsize=[8,8])
        plt.grid(True)
        plt.plot(a['Valores'],label='Valores',marker='o')
        plt.plot(a['Promedio a 3'],label='Media Móvil 3 años')
        plt.plot(a['Promedio a 4'],label='Media Móvil 4 años')
        plt.legend(loc=2)
        if not os.path.isdir('static'):
            os.mkdir('static')
        else:
            # Remove old plot files
            for filename in glob.glob(os.path.join('static', '*.png')):
                os.remove(filename)
        # Use time since Jan 1, 1970 in filename in order make
        # a unique filename that the browser has not chached
        plotfile = os.path.join('static', str(time.time()) + '.png')
        plt.savefig(plotfile)
        plt.clf()

        del df['Año']
        return render_template('/metspages/metprob/prommovil.html', form=form, tables=[df.to_html(classes='data table table-bordered')], grafica = plotfile, res1=resv1,
        res2=resv2,res3=resv3,res4=resv4,res5=resv5,res6=resv6,res7=resv7,res8=resv8)
    else:
        N = None
        resv1= None
        resv2 = None
        resv3 = None
        resv4 = None
        resv5 = None
        resv6= None
        resv7 = None
        resv8 = None
    return render_template('/metspages/metprob/prommovil.html', form=form, N=N, res1=resv1,
        res2=resv2,res3=resv3,res4=resv4,res5=resv5,res6=resv6,res7=resv7,res8=resv8)
#######------

@app.route('/alisexponencial', methods=("POST", "GET"))
def alisexponencial():

    from flask import Blueprint, Flask, render_template, make_response, request, send_file
    from wtforms import Form, FloatField, validators,StringField, IntegerField
    from numpy import exp, cos, linspace
    from math import pi
    import io
    import random
    import os, time, glob
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class InputForm(Form):
            N = StringField(
                label='Escriba los valores de ingreso separados por comas (,)', default='5, 6 ,8, 7',
                validators=[validators.InputRequired()])
            M = FloatField(
                label='Valor de alfa (entre 0 y 1)', default=0.1,
                validators=[validators.InputRequired(), validators.NumberRange(min=0.1, max=1, message='Solo valores entre 0 y 1')])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        # el DataFrame se llama movil
        M = (form.M.data)
        prueba = str(form.N.data)
        ex = prueba.split(",")
        ex2 =list(map(float,ex))
        exporta = {'Año':ex2,
        'Valores':ex2}
        movil = pd.DataFrame(exporta)
        # mostramos los 5 primeros registros
        movil.head()
        alfa = M
        unoalfa = 1. - alfa
        for i in range(0,movil.shape[0]-1):
            movil.loc[movil.index[i+1],'SN'] = np.round(movil.iloc[i,1],1)
        for i in range(2,movil.shape[0]):
            movil.loc[movil.index[i],'SN'] = np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
        i=i+1
        p2=np.round(movil.iloc[i-1,1],1)*alfa + np.round(movil.iloc[i-1,2],1)*unoalfa
        movil['Error pronóstico'] = movil['Valores']-movil['SN']

        plt.figure(figsize=[8,8])
        plt.grid(True)
        plt.plot(movil['Valores'],label='Valores originales')
        plt.plot(movil['SN'],label='Suavización Exponencial')
        plt.legend(loc=2)
        if not os.path.isdir('static'):
            os.mkdir('static')
        else:
            # Remove old plot files
            for filename in glob.glob(os.path.join('static', '*.png')):
                os.remove(filename)
        # Use time since Jan 1, 1970 in filename in order make
        # a unique filename that the browser has not chached
        plotfile = os.path.join('static', str(time.time()) + '.png')
        plt.savefig(plotfile)
        plt.clf()

        a = movil.append({'Año':2018,'Valores':"Pronóstico", 'SN':p2},ignore_index=True)
        df = a
        del df['Año']

        deftemp1 = df['Valores']
        deftemp2 = df['SN']
        deftemp3 = df['Error pronóstico']
        resv1 = deftemp1[1]
        resv2 = deftemp2[1]
        resv3 = deftemp2[2]
        resv4 = deftemp3[2]
        resv5 = deftemp1[2]
        # movil
        #%matplotlib inline
        

        return render_template('/metspages/metprob/alisexponencial.html', form=form, tables=[df.to_html(classes='data table table-bordered')],grafica = plotfile, M=alfa,res1=resv1,res2=resv2,res3=resv3,res4=resv4,res5=resv5)
    else:
        N = None
        M = None
        resv1 = None
        resv2 = None
        resv3 = None
        resv4 = None
        resv5 = None
    return render_template('/metspages/metprob/alisexponencial.html', form=form, N=N, M=M,res1=resv1,res2=resv2,res3=resv3,res4=resv4,res5=resv5)
######### Promedios fin ####

### regresion lineal inicio ### 


@app.route('/reglineal', methods=("POST", "GET"))
def reglineal():
    from flask import Blueprint, Flask, render_template, make_response, request, send_file
    from wtforms import Form, FloatField, validators,StringField, IntegerField
    from numpy import exp, cos, linspace
    from math import pi
    import io
    import random
    import os, time, glob
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class InputForm(Form):
        N = StringField(
            label='Escriba los valores de X separados por comas (,)', default='7,1,10,5,4,3,13,10,2',
            validators=[validators.InputRequired()])
        M = StringField(
            label='Escriba los valores de Y separados por comas (,)', default='2,9,2,5,7,11,2,5,14',
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        # datos experimentales
        # el DataFrame se llama movil
        prueba = str(form.N.data)
        prueba2 = str(form.M.data)
        ex = prueba.split(",")
        ex2 = prueba2.split(",")
        valX =list(map(float,ex))
        valY =list(map(float,ex2))
        exporta = {'X':valX,
        'Y':valY}

        a = pd.DataFrame(exporta)
        x = a['X']
        y= a['Y']
        df = pd.DataFrame({'X':x,'Y':y})
        x2 = df["X"]**2
        xy = df["X"] * df["Y"]
        df["X^2"] = x2
        df["XY"] = xy
        
        # ajuste de la recta (polinomio de grado 1 f(x) = ax + b)
        p = np.polyfit(x,y,1) # 1 para lineal, 2 para polinomio ...
        p0,p1 = p
        P0 = p0
        P1 = p1
        pfinal = -(p1/p0)
        y_ajuste = p[0]*x + p[1]
        df['Ajuste'] = y_ajuste

        cant = len(df['Y'])

        cant1 = df['X']
        cant2 = df['Y']
        cant3 = df['X^2']
        cant4 = df['XY']

        sum1 = cant1.values.sum()
        sum2 = cant2.values.sum()
        sum3 = cant3.values.sum()
        sum4 = cant4.values.sum()
        # dibujamos los datos experimentales de la recta
        p_datos =plt.plot(x,y,'b.')
        # Dibujamos la recta de ajuste
        p_ajuste = plt.plot(x,y_ajuste, 'r-')
        plt.title('Ajuste lineal por mínimos cuadrados')
        plt.xlabel('Eje x')
        plt.ylabel('Eje y')
        plt.legend(('Datos experimentales','Ajuste lineal',), loc="upper right")
        if not os.path.isdir('static'):
            os.mkdir('static')
        else:
            # Remove old plot files
            for filename in glob.glob(os.path.join('static', '*.png')):
                os.remove(filename)
        # Use time since Jan 1, 1970 in filename in order make
        # a unique filename that the browser has not chached
        plotfile = os.path.join('static', str(time.time()) + '.png')
        plt.savefig(plotfile)
        plt.clf()

       
        return render_template('/metspages/metreg/reglineal.html', form=form, tables=[df.to_html(classes='data table table-bordered')], grafica=plotfile, cant=cant, sum1=sum1,
        sum2=sum2,sum3=sum3,sum4=sum4,P0=P0, P1=P1,fin=pfinal)
    else:
        N = None
        M = None
        cant= None
        sum1 = None
        sum2 = None
        sum3 = None
        sum4 = None
        P0= None
        P1 = None
        fin= None
    return render_template('/metspages/metreg/reglineal.html', form=form, N=N,M=M,cant=cant,sum1=sum1,sum2=sum2,sum3=sum3,sum4=sum4,P0=P0,P1=P1,fin=fin)

#### ----Regresion lineal fin---- #########
##-----Regresion cuadratica----------### 
@app.route('/regnolineal', methods=("POST", "GET"))
def regnolineal():
    from flask import Blueprint, Flask, render_template, make_response, request, send_file
    from wtforms import Form, FloatField, validators,StringField, IntegerField
    from numpy import exp, cos, linspace
    from math import pi
    import io
    import random
    import os, time, glob
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class InputForm(Form):
        N = StringField(
            label='Escriba los valores de X separados por comas (,)', default='1850,1860,1870,1880,1890,1900,1910,1920,1930,1940,1950',
            validators=[validators.InputRequired()])
        M = StringField(
            label='Escriba los valores de Y separados por comas (,)', default='23.2,31.4,39.8,50.2,62.9,76.0,92.0,105.7,122.8,131.7,151.1',
            validators=[validators.InputRequired()])
        C = IntegerField(
            label='Dato a predecir', default=1,
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        # Importar libreria numpy
        # datos experimentales
        # el DataFrame se llama movil
        prueba = str(form.N.data)
        prueba2 = str(form.M.data)
        PRED = int(form.C.data)

        
        ex = prueba.split(",")
        ex2 = prueba2.split(",")
        valX =list(map(float,ex))
        valY =list(map(float,ex2))
        exporta = {'ValTiempo':valX,
        'Y':valY}
        a = pd.DataFrame(exporta)
        cantidad = len(a['ValTiempo'])
        c=2
        x=[0]
        ini=1
        fin=-1
        while c <= cantidad:
            if c % 2 == 0:
                x.append(ini)
                ini=ini+1
                c = c+1
            else:
                x.insert(0,fin)
                fin=fin-1
                c = c+1
        a['X'] = x
        x = a['X']
        y= a['Y']
        ValTiempo = a["ValTiempo"]
        df = pd.DataFrame({'ValTiempo':ValTiempo,'X':x,'Y':y})
        x2 = df["X"]**2
        x3 = df["X"]**3
        x4 = df["X"]**4
        xy = df["X"] * df["Y"]
        x2y = x2 * df["Y"]
        df["X^2"] = x2
        df["X^3"] = x3
        df['X^4'] = x4
        df["XY"] = xy
        df["X^2Y"] = x2y

        cant1 = df['X']
        cant2 = df['Y']
        cant3 = df['X^2']
        cant4 = df['X^3']
        cant5 = df['X^4']
        cant6 = df['XY']
        cant7 = df['X^2Y']

        sum1 = cant1.values.sum()
        sum2 = cant2.values.sum()
        sum3 = cant3.values.sum()
        sum4 = cant4.values.sum()
        sum5 = cant5.values.sum()
        sum6 = cant6.values.sum()
        sum7 = cant7.values.sum()


        p = np.polyfit(x,y,2)
        p0,p1,p2 = p
        P0 = p0
        P1 = p1
        P2 = p2
        #print ("El valor de p0 = ", p0, "Valor de p1 = ", p1, " el valor de p2 = ",p2)
        y_ajuste = p[0]*x*x + p[1]*x + p[2]
        n=x.size
        x1 = []
        x2 = []
        for i in [PRED]:
            y1_ajuste = p[0]*i*i + p[1]*i + p[2]
            x1.append(i)
            x2.append(y1_ajuste)
        df["Ajuste"]=y_ajuste
        dp = pd.DataFrame({'ValTiempo':'Dato buscado','X':PRED, 'Y':[0],'Ajuste':x2})
        res=x2[-1]
        df = df.append(dp,ignore_index=True)
        
        p_datos =plt.plot(x,y,'b.')
        # Dibujamos la curva de ajuste
        p_ajuste = plt.plot(x,y_ajuste, 'r-')
        plt.title('Ajuste Polinomial por mínimos cuadrados')
        plt.xlabel('Eje x')
        plt.ylabel('Eje y')
        plt.legend(('Datos experimentales','Ajuste Polinomial',), loc="upper left")
        if not os.path.isdir('static'):
            os.mkdir('static')
        else:
            # Remove old plot files
            for filename in glob.glob(os.path.join('static', '*.png')):
                os.remove(filename)
        # Use time since Jan 1, 1970 in filename in order make
        # a unique filename that the browser has not chached
        plotfile = os.path.join('static', str(time.time()) + '.png')
        plt.savefig(plotfile)
        plt.clf()

        
        return render_template('/metspages/metreg/regnolineal.html', form=form, tables=[df.to_html(classes='data table table-bordered')], grafica=plotfile, sum1=sum1,
        sum2=sum2,sum3=sum3,sum4=sum4,sum5=sum5,sum6=sum6,sum7=sum7,P0=P0, P1=P1,P2=P2,cant=cantidad,pron=PRED,res=res)
    else:
        N = None
        M = None
        C = None
        sum1 = None
        sum2 = None
        sum3 = None
        sum4 = None
        sum5 = None
        sum6 = None
        sum7= None
        P0= None
        P1 = None
        P2= None
        cantidad = None
        PRED= None
        res= None
    return render_template('/metspages/metreg/regnolineal.html', form=form, N=N, M=M, C=C,sum1=sum1,
        sum2=sum2,sum3=sum3,sum4=sum4,sum5=sum5,sum6=sum6,sum7=sum7,P0=P0, P1=P1,P2=P2,cant=cantidad,pron=PRED,res=res)


#### regresion lineal cudratica fin ####


#### montecarlo ####
@app.route('/montecarloaditivo', methods=("POST", "GET"))
def montecarloaditivo():
    from flask import Blueprint, Flask, render_template, make_response,request, send_file
    from wtforms import Form, FloatField, validators,StringField, IntegerField
    from numpy import exp, cos, linspace
    import math
    import itertools
    import io
    import random
    import os, time, glob
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class InputForm(Form):
        L = StringField(
            label='Escriba los valores de ingreso separados por comas (,)', default='5501.0, 6232.7, 8118.3, 10137.00, 10449.50, 12794.60, 9939.10,  13193.00, 16036.2, 18496.90, 18709.30, 19363.50, 16521.50, 15175.40,  16927.00',
            validators=[validators.InputRequired()])
        N = IntegerField(
            label='Número de eventos que desea', default=20,
            validators=[validators.InputRequired()])
        M = IntegerField(
            label='Módulo', default=1000,
            validators=[validators.InputRequired()])
        A = IntegerField(
            label='Multiplicador', default=747,
            validators=[validators.InputRequired()])
        X0 = IntegerField(
            label='Semilla', default=123,
            validators=[validators.InputRequired()])
        C = IntegerField(
            label='Incremento', default=457,
            validators=[validators.InputRequired()])

    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        # datos experimentales
        # el DataFrame se llama movil
        prueba = str(form.L.data)
        ex = prueba.split(",")
        ex2 =list(map(float,ex))
        exporta = {'Año':ex2,
        'Valores':ex2}
        a = pd.DataFrame(exporta)
        cant = len(a['Valores'])
        cantidad = list(range(cant + 1))
        cantidad[1:]
        a['Año'] = cantidad[1:]


        dfval = exporta['Valores']
        
        # Ordenamos por Día
        suma = a['Valores'].sum()
        ##cant=len(exporta)
        suma
        x1 = a.assign(Probabilidad=lambda x: x['Valores'] / suma)
        x2 = x1.sort_values('Año')

        salvando = x2['Año']
        del x2['Año']
        a=x2['Probabilidad']
        a1= np.cumsum(a) #Cálculo la suma acumulativa de las probabilidades
        x2['FPA'] =a1
        x2['Min'] = x2['FPA']
        x2['Max'] = x2['FPA']
        lis = x2["Min"].values
        lis2 = x2['Max'].values
        lis[0]= 0
        for i in range(1,len(x2['Valores'])):
            lis[i] = lis2[i-1]
        x2['Min'] = lis

        dfprob = x2['Probabilidad']

        n = int(form.N.data)
        m = int(form.M.data)
        a = int(form.A.data)
        x0 = int(form.X0.data)
        c = int(form.C.data)
        x = [1] * n
        r = [0.1] * n
        for i in range(0, n):
            x[i] = ((a*x0)+c) % m
            x0 = x[i]
            r[i] = x0 / m
        # llenamos nuestro DataFrame
        d = {'ri': r }
        dfMCL = pd.DataFrame(data=d)
        dfMCL
        max = x2 ['Max'].values
        min = x2 ['Min'].values
        def busqueda(arrmin, arrmax, valor):
        #print(valor)
            for i in range (len(arrmin)):
            # print(arrmin[i],arrmax[i])
                if valor >= arrmin[i] and valor <= arrmax[i]:
                    return i
                    #print(i)
            return -1
        xpos = dfMCL['ri']
        posi = [0] * n
        #print (n)
        for j in range(n):
            val = xpos[j]
            pos = busqueda(min,max,val)
            posi[j] = pos
        df1 = x2

        simula = []
        for j in range(n):
            for i in range(n):
                sim = x2.loc[salvando == posi[i]+1]
                simu = sim.filter(['Valores']).values
                iterator = itertools.chain(*simu)
                for item in iterator:
                    a=item
                simula.append(round(a,2))
        dfMCL["Simulación"] = pd.DataFrame(simula)
        df2 = dfMCL
        return render_template('/metspages/metsim/montecarlo.html', form=form, tables=[df1.to_html(classes='data table table-bordered')], tables2=[df2.to_html(classes='data table table-bordered')], suma=suma,vald1=dfval[0],
        cant=cant,dfprob=dfprob[0])
    else:
        N = None
        M = None
        A = None
        X0 = None
        C = None
        L = None
        grafica = None
        vald1= None
        cant= None
        suma= None
        dfprob = None
    return render_template('/metspages/metsim/montecarlo.html', form=form, L=L, N=N, M=M, A=A, X0=X0, C=C, grafica=grafica,suma=suma,vald1=vald1,cant=cant,dfprob=dfprob)

###### fin MONTECARLO  ####

#### inicio inventario EQQ ###


@app.route('/inventarioEOQ', methods=("POST", "GET"))
def inventarioEOQ():
    from flask import Blueprint, Flask, render_template, make_response, request, send_file
    from wtforms import Form, FloatField, validators,StringField, IntegerField
    from numpy import exp, cos, linspace
    from math import pi, sqrt
    import io
    import random
    import os, time, glob
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class InputForm(Form):
        D = FloatField(
            label='Valor de la demanda anual', default=12000,
            validators=[validators.InputRequired()])
        CO = FloatField(
            label='Costo de ordenar', default=25.00,
            validators=[validators.InputRequired()])
        CH = FloatField(
            label='Costo de mantenimiento', default=0.50,
            validators=[validators.InputRequired()])
        P = FloatField(
            label='Costo por unidad del producto', default=2.50,
            validators=[validators.InputRequired()])
        TE = IntegerField(
            label='Tiempo de espera del producto en días', default=5,
            validators=[validators.InputRequired()])
        DA = IntegerField(
            label='Días habíles del año', default=250,
            validators=[validators.InputRequired()])
        PE = IntegerField(
            label='Periodo del inventario', default=30,
            validators=[validators.InputRequired()])
    
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
       
        D = float(form.D.data)  #Cantidad que se tiene pa distribuir
        Co = float(form.CO.data)
        Ch = float(form.CH.data)
        P = float(form.P.data)
        Tespera = int(form.TE.data)
        DiasAno = int(form.DA.data)
        Periodo = int(form.PE.data)
        Q = round(sqrt(((2*Co*D)/Ch)),2)
        N = round(D / Q,2)
        R = round((D / DiasAno) * Tespera,2)
        T = round(DiasAno / N,2)
        CoT = N * Co
        ChT = round(Q / 2 * Ch,2)
        MOQ = round(CoT + ChT,2)
        CTT = round(P * D + MOQ,2)
        

        # Programa para generar el gráfico de costo mínimo
        indice = ['Q','Costo_ordenar','Costo_Mantenimiento','Costo_total','Diferencia_Costo_Total']
        # Generamos una lista ordenada de valores de Q

        periodo = np.arange(1,Periodo)
        def genera_lista(Q):
            n= Periodo-1
            Q_Lista = []
            i=1
            Qi = Q
            Q_Lista.append(Qi)
            for i in range(1,int(Periodo/2)):
                Qi = Qi - 60
                Q_Lista.append(Qi)

            Qi = Q
            for i in range(int(Periodo/2), n):
                Qi = Qi + 60
                Q_Lista.append(Qi)

            return Q_Lista
        Lista= genera_lista(Q)
        Lista.sort()
        dfQ = pd.DataFrame(index=periodo, columns=indice).fillna(0)
        dfQ['Q'] = Lista
        #dfQ
        for period in periodo:
            dfQ['Costo_ordenar'][period] = D * Co / dfQ['Q'][period]
            dfQ['Costo_Mantenimiento'][period] = dfQ['Q'][period] * Ch / 2
            dfQ['Costo_total'][period] = dfQ['Costo_ordenar'][period] + dfQ['Costo_Mantenimiento'][period]
            dfQ['Diferencia_Costo_Total'][period] = dfQ['Costo_total'][period] - MOQ
        pd.set_option('mode.chained_assignment', None)
        df = dfQ


        dfG = dfQ.loc[:,'Costo_ordenar':'Costo_total']
        dfG
        dfG.plot()

        if not os.path.isdir('static'):
            os.mkdir('static')
        else:
            # Remove old plot files
            for filename in glob.glob(os.path.join('static', '*.png')):
                os.remove(filename)
        # Use time since Jan 1, 1970 in filename in order make
        # a unique filename that the browser has not chached
        plotfile = os.path.join('static', str(time.time()) + '.png')
        plt.savefig(plotfile)
        plt.clf()

       
        return render_template('/metspages/modsim/inventeoq.html', form=form, tables=[df.to_html(classes='data table table-bordered')], grafica=plotfile, dato1=Q,
        dato2=CoT,dato3=ChT,dato4=MOQ,dato5=CTT,dato6=N,dato7=R,dato8=T)
    else:
        D = None
        CO = None
        CH= None
        P = None
        TE = None
        DA = None
    return render_template('/metspages/modsim/inventeoq.html', form=form, D=D,CO=CO,CH=CH,P=P,TE=TE,DA=DA)

### fiin inventario


@app.route('/lineaEspera')
def lineaEspera():
    return render_template('lineaEspera.html')

@app.route('/imprimirLineaEspera')
def imprimirLineaEspera():
    return render_template('imprimirLineaEspera.html')

@app.route('/calcularLineaEspera', methods=['GET','POST'])
def calcularLineaEspera():
    landa = request.form.get("landa", type=float)
    nu = request.form.get("miu", type=float)
    num = request.form.get("numeroIteraciones", type=int)
    x0 = request.form.get("semilla", type=int) 
    a = request.form.get("multiplicador", type=int)
    c = request.form.get("incremento", type=int)
    m = request.form.get("modulo", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO 
    import base64
    import math, random
    from pandas import DataFrame

    #La probabilidad de hallar el sistema ocupado o utilización del sistema:
    p=[]
    p=landa/nu
    #La probabilidad de que no haya unidades en el sistema este vacía u ocioso :
    Po=[]
    Po = 1.0 - (landa/nu)
    #Longitud esperada en cola, promedio de unidades en la línea de espera:
    Lq=[]
    Lq = landa*landa / (nu * (nu - landa))
    #/ (nu * (nu - landa))
    # Número esperado de clientes en el sistema(cola y servicio) :
    L=[]
    L = landa /(nu - landa)
    #El tiempo promedio que una unidad pasa en el sistema:
    W=[]
    W = 1 / (nu - landa)
    #Tiempo de espera en cola:
    Wq=[]
    Wq = W - (1.0 / nu)
    print (Wq)
    #La probabilidad de que haya n unidades en el sistema:
    n= 1
    Pn=[]
    Pn = (landa/nu)*n*Po


    df = pd.DataFrame(columns=('landa', 'nu', 'p', 'Po', 'Lq', 'L', 'W', 'Wq', 'Pn'))
    df.loc[len(df)]=[landa, nu, p, Po, Lq, L, W, Wq, Pn] 
    df

    data= df.to_html(classes="table table-light table-striped", justify="justify-all", border=0)

    i = 0
    # Landa y nu ya definidos
    # Atributos del DataFrame
    """
    ALL # ALEATORIO DE LLEGADA DE CLIENTES
    ASE # ALEATORIO DE SERVICIO
    TILL TIEMPO ENTRE LLEGADA
    TISE TIEMPO DE SERVICIO
    TIRLL TIEMPO REAL DE LLEGADA
    TIISE TIEMPO DE INICIO DE SERVICIO
    TIFSE TIEMPO FINAL DE SERVICIO
    TIESP TIEMPO DE ESPERA
    TIESA TIEMPO DE SALIDA
    numClientes NUMERO DE CLIENTES
    dfLE DATAFRAME DE LA LINEA DE ESPERA
    """
    numClientes=num
    i = 0
    indice = ['ALL','ASE','TILL','TISE','TIRLL','TIISE','TIFSE','TIESP','TIESA']
    Clientes = np.arange(numClientes)
    dfLE = pd.DataFrame(index=Clientes, columns=indice).fillna(0.000)
    
    #np.random.seed(num)

    # n, m, a, x0 = 20, 1000, 747, 123
    x = [1] * num
    r = [0.1] * num
    for j in range(0, num):
        x[j] = ((a*x0)+c) % m
        x0 = x[j]
        #r[j] = x0 / m
        dfLE['ALL'][j] = x0 / m
        #dfLE['ASE'][j] = x0 / m

    # n, m, a, x0 = 20, 1000, 747, 123
    x = [1] * num
    r = [0.1] * num
    for j in range(0, num):
        x[j] = (a*x0) % m
        x0 = x[j]
        #r[j] = x0 / m
        #dfLE['ALL'][j] = x0 / m
        dfLE['ASE'][j] = x0 / m

    for i in Clientes:
        if i == 0:
            #dfLE['ASE'][i] = random.random()
            dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
            dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
            dfLE['TIRLL'][i] = dfLE['TILL'][i]
            dfLE['TIISE'][i] = dfLE['TIRLL'][i]
            dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
            dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
        else:
            #dfLE['ASE'][i] = random.random()
            dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
            dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
            dfLE['TIRLL'][i] = dfLE['TILL'][i] + dfLE['TIRLL'][i-1]
            dfLE['TIISE'][i] = max(dfLE['TIRLL'][i],dfLE['TIFSE'][i-1])
            dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
            dfLE['TIESP'][i] = dfLE['TIISE'][i] - dfLE['TIRLL'][i]
            dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
    nuevas_columnas = pd.core.indexes.base.Index(["A_LLEGADA","A_SERVICIO","TIE_LLEGADA","TIE_SERVICIO",
     "TIE_EXACTO_LLEGADA","TIE_INI_SERVICIO","TIE_FIN_SERVICIO",
     "TIE_ESPERA","TIE_EN_SISTEMA"])

    dfLE.columns = nuevas_columnas
    dfLE

       # Graficamos los numeros generados
    buf = io.BytesIO()
    plt.plot(dfLE['A_LLEGADA'],label='A_LLEGADA')
    plt.plot(dfLE['A_SERVICIO'],label='A_SERVICIO')
    plt.plot(dfLE['TIE_LLEGADA'],label='TIE_LLEGADA')
    plt.plot(dfLE['TIE_SERVICIO'],label='TIE_SERVICIO')
    plt.plot(dfLE['TIE_EXACTO_LLEGADA'],label='TIE_EXACTO_LLEGADA')
    plt.plot(dfLE['TIE_INI_SERVICIO'],label='TIE_INI_SERVICIO')
    plt.plot(dfLE['TIE_FIN_SERVICIO'],label='TIE_FIN_SERVICIO')
    plt.plot(dfLE['TIE_ESPERA'],label='TIE_ESPERA')
    plt.plot(dfLE['TIE_EN_SISTEMA'],label='TIE_EN_SISTEMA')
    plt.legend(loc=2)
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')


    kl=dfLE["TIE_ESPERA"]
    jl=dfLE["TIE_EN_SISTEMA"]
    ll=dfLE["A_LLEGADA"]
    pl=dfLE["A_SERVICIO"]
    ml=dfLE["TIE_INI_SERVICIO"]
    nl=dfLE["TIE_FIN_SERVICIO"]

    klsuma=sum(kl)
    klpro=(klsuma/num)
    jlsuma=sum(jl)
    jlpro=jlsuma/num
    dfLE.loc[num]=['-','-','-','-','-','-','SUMA',klsuma,jlsuma]
    dfLE.loc[(num+1)]=['-','-','-','-','-','-','PROMEDIO',klpro,jlpro]

    dfLE

    data2= dfLE.to_html(classes="table table-light table-striped", justify="justify-all", border=0)
    
    writer = ExcelWriter("static/file/data.xlsx")
    dfLE.to_excel(writer, index=False)
    writer.save()  

    dfLE.to_csv("static/file/data.csv", index=False)

    dfLE2 = pd.DataFrame(dfLE.describe())
    data3 = dfLE2.to_html(classes="table table-light table-striped", justify="justify-all", border=0)

    return render_template('imprimirLineaEspera.html', data=data, data2=data2, data3=data3 ,image=plot_url)

@app.route('/calcularMED', methods=['GET','POST'])
def calcularMED():
    columna = request.form.get("nombreColumna")

    file = request.files['file'].read()

    # importamos la libreria Pandas, matplotlib y numpy que van a ser de mucha utilidad para poder hacer gráficos
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO 
    import base64
    from pandas import DataFrame

    # leemos los datos de la tabla del directorio Data de trabajo
    datos = pd.read_excel(file)
    #Presentamos los datos en un DataFrame de Pandas
    #datos.to_html()
    datohtml = datos.to_html(classes="table table-light table-striped", justify="justify-all", border=0)
    # Preparando para el grafico para la columna TOTAL PACIENTES
    buf = io.BytesIO()
    x=datos[columna]
    plt.figure(figsize=(10,5))
    plt.hist(x,bins=8,color='blue')
    plt.axvline(x.mean(),color='red',label='Media')
    plt.axvline(x.median(),color='yellow',label='Mediana')
    plt.axvline(x.mode()[0],color='green',label='Moda')
    plt.xlabel('Total de datos')
    plt.ylabel('Frecuencia')
    plt.legend()
    
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    media = datos[columna].mean()
    moda = datos[columna].mode()
    mediana = datos[columna].median()
    
    df = pd.DataFrame(columns=('Media', 'Moda', 'Mediana'))
    df.loc[len(df)]=[media, moda, mediana] 
    df
    data = df.to_html(classes="table table-light table-striped", justify="justify-all", border=0)

    # Tomamos los datos de las columnas
    df2 = datos[[columna]].describe()
    # describe(), nos presenta directamente la media, desviación standar, el valor mínimo, valor máximo, el 1er cuartil, 2do Cuartil, 3er Cuartil
    data2 = df2.to_html(classes="table table-light table-striped", justify="justify-all", border=0)

    return render_template('imprimirMED.html', datohtml=datohtml, data=data, data2=data2, image=plot_url)


if __name__ == '__main__':
    app.run(debug=True)