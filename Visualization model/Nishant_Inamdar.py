
# Importing Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages
import sys


# Importing Dataset from CommandLine
val = sys.argv[1]
dataset = pd.read_csv(val)

#dataset.drop(["Certifications/Achievement/ Research papers", "Link to updated Resume (Google/ One Drive link preferred)","link to Linkedin profile"], axis = 1, inplace = True)
df = pd.DataFrame(dataset)

# Label Encoding of Necessary Data
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['college'] = label.fit_transform(df['College name'])
df['Areas'] = label.fit_transform(df['Areas of interest'])
df['Standard'] = label.fit_transform(df['Which-year are you studying in?'])
df['gender'] = label.fit_transform(df['Gender'])
df['major'] = label.fit_transform(df['Major/Area of Study'])
df['Creteria'] = label.fit_transform(df['Label'])
df.head(5)

# Defining Necessary Functions
#def make_autopct(values):
#    def my_autopct(pct):
#        total = sum(values)
#        val = int(round(pct*total/100.0))
#        return '{v:d}'.format(p=pct,v=val)
#    return my_autopct

def showtext(bars):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+ .05, yval + 5, yval)
    return

def abc():
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.7, 0.8), shadow=True, ncol=1)
    return

with PdfPages('visualization-output.pdf') as pdf:
    
    #a. The number of students applied to different technologies
    interest = df['Areas of interest'].value_counts()
    values = interest.tolist()
    labels = interest.index

    fig = plt.figure()
    sns.set(font_scale = 1.5)
    fig_dims = (15, 12)
    fig, ax = plt.subplots(figsize=fig_dims)
    plt.title('Areas of Interest\n\n\n')
    plt.pie(interest,labels = labels,startangle=140,shadow = True,wedgeprops = {'linewidth': 10},autopct='%2.2f%%')
    plt.axis('equal')
    #plt.show()
    pdf.savefig(fig)
    plt.close()
    
#------------------------------------------------------------------------------
    
    #b. The number of students applied for Data Science who knew ‘’Python” and who didn’t.
    pythonbased = df.loc[(df['Areas']==5) & (df['Programming Language Known other than Java (one major)']=='Python')].shape[0]
    nonpythonbased = df.loc[(df['Areas']==5) & (df['Programming Language Known other than Java (one major)']!='Python')].shape[0]

    fig2 = plt.figure(figsize = (15,12))
    plt.grid(True)
    sns.set(font_scale = 1.5)
    bars = plt.bar(['Knows Python','Does not know Python'],[pythonbased,nonpythonbased],color = ['blue','green'],width = 0.30)
    plt.yticks(np.arange(0,600,25))
    plt.title("The number of students applied for Data Science\n who knew ‘’Python” and who didn’t.\n")
    plt.xlabel("Categories")
    plt.ylabel("Number of Students")
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+ .1, yval + 1, yval)
    #plt.tight_layout()
    #plt.show()
    pdf.savefig(fig2)
    plt.close()
    
#------------------------------------------------------------------------------

    #c. The different ways students learned about this program.   
    source=df['How Did You Hear About This Internship?'].value_counts()
    values2 = source.tolist()
    fields = source.index
    
    fig = plt.figure()
    fig_dims = (15, 12)
    fig, ax = plt.subplots(figsize=fig_dims)
    plt.title('The different ways students learned about this program.\n')
    plt.pie(values2,labels =fields ,startangle=140,shadow = True,wedgeprops = {'linewidth': 10},autopct='%2.2f%%')
    plt.axis('equal')
    #plt.show()
    pdf.savefig(fig)
    plt.close()
    
#------------------------------------------------------------------------------

    #d. Students who are in the fourth year and have a CGPA greater than 8.0    
    fpass = df.loc[(df['Standard']==1) & (df['CGPA/ percentage']>8.0)]
    ffail = df.loc[(df['Standard']==1) & (df['CGPA/ percentage']<=8.0)]

    fig4 = plt.figure(figsize = (15,12))
    plt.grid(True)
    bars = plt.bar(['Above 8.0 CGPA','Below 8.0 CGPA'],[fpass.shape[0],ffail.shape[0]],color = ['blue','green'],width = 0.30)
    plt.yticks(np.arange(0,2000,100))
    plt.title("Students who are in the fourth year \nand have a CGPA greater than 8.0.\n") 
    plt.xlabel("Categories")
    plt.ylabel("Number of Students")
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+ .1, yval + 5, yval)
    #plt.show()
    pdf.savefig(fig4)
    plt.close()    
    
#------------------------------------------------------------------------------

    #e. Students who applied for Digital Marketing with verbal and written communication score greater than 8.
    pdm = df.loc[(df['Areas']==7) & (df['Rate your written communication skills [1-10]']>8.0) & ((df['Rate your verbal communication skills [1-10]']>8.0))]
    fdm = df.loc[(df['Areas']==7) & (df['Rate your written communication skills [1-10]']<=8.0) & ((df['Rate your verbal communication skills [1-10]']<=8.0))]

    fig5 = plt.figure(figsize = (15,12))
    plt.grid(True)
    bars = plt.bar(['Above 8.0','Below 8.0'],[pdm.shape[0],fdm.shape[0]],color = ['blue','green'],width = 0.30)
    plt.yticks(np.arange(0,225,25))
    plt.title("Students who applied for Digital Marketing with verbal\n and written communication score greater than 8.0 \n") 
    plt.xlabel("Categories")
    plt.ylabel("Number of Students")
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+ .1, yval + 3, yval)
    plt.tight_layout()
    #plt.show()
    pdf.savefig(fig5)
    plt.close()
    
#------------------------------------------------------------------------------

    #f. Year-wise and area of study wise classification of students.
    Subject=df['Major/Area of Study']
    Subject.value_counts()
    #['Computer Engineering' 'Electrical Engineering''Electronics and Telecommunication']
    comp1 = df.loc[(df['major']==0) & (df['Standard']==0)] # for first year
    comp2 = df.loc[(df['major']==0) & (df['Standard']==2)] # for second year
    comp3 = df.loc[(df['major']==0) & (df['Standard']==3)] # for third year
    comp4 = df.loc[(df['major']==0) & (df['Standard']==1)] # for fourth year
    #----------------------------------------------------------------------------------------------------------
    elec1 = df.loc[(df['major']==1) & (df['Standard']==0)] # for first year
    elec2 = df.loc[(df['major']==1) & (df['Standard']==2)] # for second year
    elec3 = df.loc[(df['major']==1) & (df['Standard']==3)] # for third year
    elec4 = df.loc[(df['major']==1) & (df['Standard']==1)] # for fourth year
    #----------------------------------------------------------------------------------------------------------
    entc1 = df.loc[(df['major']==2) & (df['Standard']==0)] # for first year
    entc2 = df.loc[(df['major']==2) & (df['Standard']==2)] # for second year
    entc3 = df.loc[(df['major']==2) & (df['Standard']==3)] # for third year
    entc4 = df.loc[(df['major']==2) & (df['Standard']==1)] # for fourth year

    comp = [comp1.shape[0],comp2.shape[0],comp3.shape[0],comp4.shape[0]]
    elec = [elec1.shape[0],elec2.shape[0],elec3.shape[0],elec4.shape[0]]
    entc = [entc1.shape[0],entc2.shape[0],entc3.shape[0],entc4.shape[0]]

    fig = plt.figure()
    fig_dims = (15, 12)
    fig, ax = plt.subplots(figsize=fig_dims)
    n_groups = 4
    index = np.arange(n_groups)
    bar_width = 0.30
    opacity = 0.8

    computer = plt.bar(index, comp, bar_width,alpha=opacity,color='blue',label='Computer Department')

    electrical = plt.bar(index + bar_width, elec, bar_width,alpha=opacity,color='green',label='Electrical Department')

    electronics = plt.bar(index + (bar_width*2) ,entc, bar_width,alpha=opacity,color='red',label='Electronics and Telecommunication')

    plt.xlabel('Standard')
    plt.ylabel('No of Students')
    plt.title('Year-wise and area of study wise classification of students.\n\n')
    plt.xticks(index + bar_width, ('First Year', 'Second Year', 'Third Year', 'Fourth Year'))
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.8), shadow=True, ncol=1)

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    showtext(computer)
    showtext(electrical)
    showtext(electronics)
    plt.tight_layout()
    #plt.show()
    pdf.savefig(fig)
    plt.close()
    
#------------------------------------------------------------------------------

    #g. City and college wise classification of students.    
    citycount=df['City'].value_counts()
    citycount.sort_values(inplace=True)
    collegecount=df['College name'].value_counts()
    collegecount.sort_values(inplace=True)

    fig = plt.figure()
    fig_dims = (20, 15)
    sns.set(font_scale = 1.5)
    fig, ax = plt.subplots(figsize=fig_dims)
    plt.title('College wise classification of students \n\n')
    plt.pie(collegecount,labels =collegecount.index,startangle=70,shadow = True,counterclock=False,wedgeprops = dict(width=0.6),pctdistance = 0.8,autopct='%2.2f%%')
    plt.axis('equal')
    #plt.show()
    pdf.savefig(fig)
    plt.close()
    #fig.savefig('C:/Users/Nishant/Documents/Datasets/Project/Final deliverables/pdfpages/image7.png')
    #print('\n\n===============================================================================================================================\n\n')
    fig = plt.figure()
    fig_dims = (15, 12)
    sns.set(font_scale = 1.5)
    fig, ax = plt.subplots(figsize=fig_dims)
    plt.title('City wise classification of students \n\n')
    plt.pie(citycount,labels =citycount.index,startangle=10,shadow = True,counterclock=False,wedgeprops ={'linewidth': 5} ,pctdistance = 0.6,autopct='%2.2f%%')
    plt.axis('equal')
    #plt.show()
    pdf.savefig(fig)
    plt.close()
    
#------------------------------------------------------------------------------

    #h. Plot the relationship between the CGPA and the target variable.    
    X = df['CGPA/ percentage'].value_counts()
    X.sort_index()
    Y=df['Creteria'].value_counts()

    p1 = df.loc[(df['CGPA/ percentage']>9.5) & (df['CGPA/ percentage']<=10.0)& (df['Creteria']==0)] 
    p2 = df.loc[(df['CGPA/ percentage']>9.0) & (df['CGPA/ percentage']<=9.5) & (df['Creteria']==0)] 
    p3 = df.loc[(df['CGPA/ percentage']>8.5) & (df['CGPA/ percentage']<=9.0) & (df['Creteria']==0)] 
    p4 = df.loc[(df['CGPA/ percentage']>8.0) & (df['CGPA/ percentage']<=8.5) & (df['Creteria']==0)] 
    p5 = df.loc[(df['CGPA/ percentage']>7.5) & (df['CGPA/ percentage']<=8.0) & (df['Creteria']==0)] 
    p6 = df.loc[(df['CGPA/ percentage']>7.0) & (df['CGPA/ percentage']<=7.5) & (df['Creteria']==0)] 

    pas = [p1.shape[0],p2.shape[0],p3.shape[0],p4.shape[0],p5.shape[0],p6.shape[0]]

    f1 = df.loc[(df['CGPA/ percentage']>9.5) & (df['CGPA/ percentage']<=10.0)& (df['Creteria']==1)] 
    f2 = df.loc[(df['CGPA/ percentage']>9.0) & (df['CGPA/ percentage']<=9.5) & (df['Creteria']==1)] 
    f3 = df.loc[(df['CGPA/ percentage']>8.5) & (df['CGPA/ percentage']<=9.0) & (df['Creteria']==1)] 
    f4 = df.loc[(df['CGPA/ percentage']>8.0) & (df['CGPA/ percentage']<=8.5) & (df['Creteria']==1)] 
    f5 = df.loc[(df['CGPA/ percentage']>7.5) & (df['CGPA/ percentage']<=8.0) & (df['Creteria']==1)] 
    f6 = df.loc[(df['CGPA/ percentage']>7.0) & (df['CGPA/ percentage']<=7.5) & (df['Creteria']==1)]

    fail = [f1.shape[0],f2.shape[0],f3.shape[0],f4.shape[0],f5.shape[0],f6.shape[0]]

    fig = plt.figure()
    fig_dims = (15, 12)
    fig, ax = plt.subplots(figsize=fig_dims)
    n_groups = 6
    index = np.arange(n_groups)
    bar_width = 0.30
    opacity = 0.8

    ele = plt.bar(index, pas, bar_width,alpha=opacity,color='blue',label='Students Eligible with Cgpa')

    nonele = plt.bar(index + bar_width, fail, bar_width,alpha=opacity,color='green',label='Students Not Eligible with Cgpa')

    plt.xlabel('Cgpa Ranges')
    plt.ylabel('No of Students')
    plt.title('Relation between the CGPA and the target variable.\n')
    plt.xticks(index + bar_width, ('9.5 to 10.0','9.0 to 9.5','8.5 to 9.0','8.0 to 8.5','7.5 to 8.0','7.0 to 7.5'))
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.2, 0.9), shadow=True, ncol=1)

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    showtext(ele)
    showtext(nonele)
    plt.tight_layout()
    #plt.show()
    pdf.savefig(fig)
    plt.close()
    
#------------------------------------------------------------------------------

    #i. Plot the relationship between the Area of Interest and the target variable.    
    X = df['Areas of interest'].value_counts()
    x=X.sort_index()
    x.index
    Y=df['Creteria'].value_counts()
    Y

    pa0 = df.loc[(df['Areas']==0) & (df['Creteria']==0)].shape[0]
    pa1 = df.loc[(df['Areas']==1) & (df['Creteria']==0)].shape[0]
    pa2 = df.loc[(df['Areas']==2) & (df['Creteria']==0)].shape[0]
    pa3 = df.loc[(df['Areas']==3) & (df['Creteria']==0)].shape[0]
    pa4 = df.loc[(df['Areas']==4) & (df['Creteria']==0)].shape[0]
    pa5 = df.loc[(df['Areas']==5) & (df['Creteria']==0)].shape[0]
    pa6 = df.loc[(df['Areas']==6) & (df['Creteria']==0)].shape[0]
    pa7 = df.loc[(df['Areas']==7) & (df['Creteria']==0)].shape[0]
    pa8 = df.loc[(df['Areas']==8) & (df['Creteria']==0)].shape[0]
    pa9 = df.loc[(df['Areas']==9) & (df['Creteria']==0)].shape[0]
    pa10 = df.loc[(df['Areas']==10) & (df['Creteria']==0)].shape[0]
    pa11 = df.loc[(df['Areas']==11) & (df['Creteria']==0)].shape[0]
    pa12 = df.loc[(df['Areas']==12) & (df['Creteria']==0)].shape[0]
    pa13 = df.loc[(df['Areas']==13) & (df['Creteria']==0)].shape[0]
    pa14 = df.loc[(df['Areas']==14) & (df['Creteria']==0)].shape[0]
    pa15 = df.loc[(df['Areas']==15) & (df['Creteria']==0)].shape[0]

    totpas = [pa0,pa1,pa2,pa3,pa4,pa5,pa6,pa7,pa8,pa9,pa10,pa11,pa12,pa13,pa14,pa15]

    fa0 = df.loc[(df['Areas']==0) & (df['Creteria']==1)].shape[0]
    fa1 = df.loc[(df['Areas']==1) & (df['Creteria']==1)].shape[0]
    fa2 = df.loc[(df['Areas']==2) & (df['Creteria']==1)].shape[0]
    fa3 = df.loc[(df['Areas']==3) & (df['Creteria']==1)].shape[0]
    fa4 = df.loc[(df['Areas']==4) & (df['Creteria']==1)].shape[0]
    fa5 = df.loc[(df['Areas']==5) & (df['Creteria']==1)].shape[0]
    fa6 = df.loc[(df['Areas']==6) & (df['Creteria']==1)].shape[0]
    fa7 = df.loc[(df['Areas']==7) & (df['Creteria']==1)].shape[0]
    fa8 = df.loc[(df['Areas']==8) & (df['Creteria']==1)].shape[0]
    fa9 = df.loc[(df['Areas']==9) & (df['Creteria']==1)].shape[0]
    fa10 = df.loc[(df['Areas']==10) & (df['Creteria']==1)].shape[0]
    fa11 = df.loc[(df['Areas']==11) & (df['Creteria']==1)].shape[0]
    fa12 = df.loc[(df['Areas']==12) & (df['Creteria']==1)].shape[0]
    fa13 = df.loc[(df['Areas']==13) & (df['Creteria']==1)].shape[0]
    fa14 = df.loc[(df['Areas']==14) & (df['Creteria']==1)].shape[0]
    fa15 = df.loc[(df['Areas']==15) & (df['Creteria']==1)].shape[0]

    totfail = [fa0,fa1,fa2,fa3,fa4,fa5,fa6,fa7,fa8,fa9,fa10,fa11,fa12,fa13,fa14,fa15]

    sns.set(font_scale = 1)
    fig = plt.figure()
    fig_dims = (15,12)
    fig, ax = plt.subplots(figsize=fig_dims)
    index = np.arange(len(x))
    bar_width = 0.30
    opacity = 0.8

    tele = ax.barh(index, totpas, bar_width,alpha=opacity,color='blue',label='Eligible with Areas of Interest')

    tnoele = ax.barh(index + bar_width, totfail, bar_width,alpha=opacity,color='green',label='Not Eligible with Areas of Interest')

    plt.xlabel('No of Students')
    plt.ylabel('Areas of Interest')
    plt.title('Relation between the Area of Interest and the target variable.\n')
    plt.yticks(index + bar_width, x.index)
    ax.invert_yaxis()
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.92, 0.8), shadow=True, ncol=1)

    plt.tight_layout()
    #plt.show()
    pdf.savefig(fig)
    plt.close()
    
#------------------------------------------------------------------------------

    #j. Plot the relationship between the year of study, major, and the target variable.
    pcomp1 = df.loc[(df['major']==0) & (df['Standard']==0) &(df['Creteria']==0)].shape[0] # for first year
    pcomp2 = df.loc[(df['major']==0) & (df['Standard']==2) &(df['Creteria']==0)].shape[0] # for second year
    pcomp3 = df.loc[(df['major']==0) & (df['Standard']==3) &(df['Creteria']==0)].shape[0] # for third year
    pcomp4 = df.loc[(df['major']==0) & (df['Standard']==1) &(df['Creteria']==0)].shape[0] # for fourth year

    fcomp1 = df.loc[(df['major']==0) & (df['Standard']==0) &(df['Creteria']==1)].shape[0] # for first year
    fcomp2 = df.loc[(df['major']==0) & (df['Standard']==2) &(df['Creteria']==1)].shape[0] # for second year
    fcomp3 = df.loc[(df['major']==0) & (df['Standard']==3) &(df['Creteria']==1)].shape[0] # for third year
    fcomp4 = df.loc[(df['major']==0) & (df['Standard']==1) &(df['Creteria']==1)].shape[0] # for fourth year

    pelec1 = df.loc[(df['major']==1) & (df['Standard']==0) &(df['Creteria']==0)].shape[0] # for first year
    pelec2 = df.loc[(df['major']==1) & (df['Standard']==2) &(df['Creteria']==0)].shape[0] # for second year
    pelec3 = df.loc[(df['major']==1) & (df['Standard']==3) &(df['Creteria']==0)].shape[0] # for third year
    pelec4 = df.loc[(df['major']==1) & (df['Standard']==1) &(df['Creteria']==0)].shape[0] # for fourth year

    felec1 = df.loc[(df['major']==1) & (df['Standard']==0) &(df['Creteria']==1)].shape[0] # for first year
    felec2 = df.loc[(df['major']==1) & (df['Standard']==2) &(df['Creteria']==1)].shape[0] # for second year
    felec3 = df.loc[(df['major']==1) & (df['Standard']==3) &(df['Creteria']==1)].shape[0] # for third year
    felec4 = df.loc[(df['major']==1) & (df['Standard']==1) &(df['Creteria']==1)].shape[0] # for fourth year

    pentc1 = df.loc[(df['major']==2) & (df['Standard']==0) &(df['Creteria']==0)].shape[0] # for first year
    pentc2 = df.loc[(df['major']==2) & (df['Standard']==2) &(df['Creteria']==0)].shape[0] # for second year
    pentc3 = df.loc[(df['major']==2) & (df['Standard']==3) &(df['Creteria']==0)].shape[0] # for third year
    pentc4 = df.loc[(df['major']==2) & (df['Standard']==1) &(df['Creteria']==0)].shape[0] # for fourth year

    fentc1 = df.loc[(df['major']==2) & (df['Standard']==0) &(df['Creteria']==1)].shape[0] # for first year
    fentc2 = df.loc[(df['major']==2) & (df['Standard']==2) &(df['Creteria']==1)].shape[0] # for second year
    fentc3 = df.loc[(df['major']==2) & (df['Standard']==3) &(df['Creteria']==1)].shape[0] # for third year
    fentc4 = df.loc[(df['major']==2) & (df['Standard']==1) &(df['Creteria']==1)].shape[0] # for fourth year

    comppass = [pcomp1,pcomp2,pcomp3,pcomp4]
    compfail = [fcomp1,fcomp2,fcomp3,fcomp4]
    elecpass = [pelec1,pelec2,pelec3,pelec4]
    elecfail = [felec1,felec2,felec3,felec4]
    entcpass = [pentc1,pentc2,pentc3,pentc4]
    entcfail = [fentc1,fentc2,fentc3,fentc4]

    sns.set(font_scale = 1.5)
    fig_dims = (15, 12)
    fig, ax = plt.subplots(figsize=fig_dims)
    n_groups = 4
    index = np.arange(n_groups)
    bar_width = 0.30
    opacity = 0.8

    compp = plt.bar(index, comppass, bar_width,alpha=opacity,color='blue',label='Elegible from Computer Department')

    compf = plt.bar(index + bar_width, compfail, bar_width,alpha=opacity,color='green',label='Not Elegible from Computer Department')

    plt.xlabel('Computer Department Classes')
    plt.ylabel('No of Students')
    plt.title('Relationship between the year of study, major\n and the target variable (Computer Department)\n')
    plt.xticks(index + bar_width, ('First Year', 'Second Year', 'Third Year', 'Fourth Year'))
    abc()
    showtext(compp)
    showtext(compf)
    plt.tight_layout()
    #plt.show()
    pdf.savefig(fig)
    plt.close()
    #fig.savefig('C:/Users/Nishant/Documents/Datasets/Project/Final deliverables/pdfpages/image11.png')
    #print("\n=========================================================================================================\n")
    fig_dims = (15, 12)
    fig, ax = plt.subplots(figsize=fig_dims)
    n_groups = 4
    index = np.arange(n_groups)
    bar_width = 0.30
    opacity = 0.8

    elecp = plt.bar(index, elecpass, bar_width,alpha=opacity,color='orange',label='Elegible from Electrical Department')

    elecf = plt.bar(index + bar_width, elecfail, bar_width,alpha=opacity,color='purple',label='Not Elegible from Electrical Department')

    plt.xlabel('Electrical Department Classes')
    plt.ylabel('No of Students')
    plt.title('Relationship between the year of study, major\n and the target variable (Electrical Department)\n')
    plt.xticks(index + bar_width, ('First Year', 'Second Year', 'Third Year', 'Fourth Year'))
    abc()
    showtext(elecp)
    showtext(elecf)
    plt.tight_layout()
    #plt.show()
    pdf.savefig(fig)
    plt.close()
    #fig.savefig('C:/Users/Nishant/Documents/Datasets/Project/Final deliverables/pdfpages/image12.png')
    #print("\n=========================================================================================================\n")

    fig_dims = (15, 12)
    fig, ax = plt.subplots(figsize=fig_dims)
    n_groups = 4
    index = np.arange(n_groups)
    bar_width = 0.30
    opacity = 0.8

    entcp = plt.bar(index, entcpass, bar_width,alpha=opacity,color='red',label='Elegible from Electronics Department')

    entcf = plt.bar(index + bar_width, entcfail, bar_width,alpha=opacity,color='black',label='Not Elegible from Electronics Department')

    plt.xlabel('Electronics Department Classes')
    plt.ylabel('No of Students')
    plt.title('Relationship between the year of study, major\n and the target variable (Electronics Department)\n')
    plt.xticks(index + bar_width, ('First Year', 'Second Year', 'Third Year', 'Fourth Year'))
    abc()
    showtext(entcp)
    showtext(entcf)
    plt.tight_layout()
    #plt.show()
    pdf.savefig(fig)
    plt.close()
    #fig.savefig('C:/Users/Nishant/Documents/Datasets/Project/Final deliverables/pdfpages/image13.png')
    #pdf.close()