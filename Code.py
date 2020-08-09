from tkinter import *
from dataFile import *
# ------------------------------------------------------------------------------------------------------

def RandomForest():

    #Model Preparation
    from sklearn.ensemble import RandomForestClassifier

    clf3 = RandomForestClassifier()
    clf3 = clf3.fit(X,np.ravel(y))

    # calculating accuracy--------------------------------------------------->
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print("Random Forest Accuracy:")
    print(accuracy_score(y_test, y_pred))

    #GUI Functionality  ----------------------------------------------------->

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0, len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z== l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0, len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")

#ExtraTree Classifier ---------------------------------------------------------------->
def extraTree():

   #Model Preparation
    from sklearn.ensemble import ExtraTreesClassifier
    clf4 = ExtraTreesClassifier(n_estimators=10,max_depth=10,min_samples_split=2,random_state=45)
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy----------------------------------------------------------------->
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print("ExtraTree Classifier accuracy:")
    print(accuracy_score(y_test, y_pred))

    # GUI Functionality-------------------------------------------------------------------->

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")

#Naive Bayes Classifier ------------------------------------------------------------------>

def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy------------------------------------------------------------------->
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print("Naive Baye's accuracy Score:")
    print(accuracy_score(y_test, y_pred))

    #GUI Functionality ---------------------------------------------------------------------->

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")

#SVM Classifier --------------------------------------------------->

def SVMFun():
    from sklearn.svm import SVC
    svclassifier = SVC(kernel = 'rbf')
    svclassifier.fit(X,np.ravel(y))

    #Calculating Accuracy Score---------------->

    from sklearn.metrics import accuracy_score,confusion_matrix
    y_pred = svclassifier.predict(X_test)
    print("SVM Accuracy Score:")
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred))

    #GUI Functionality--------------------------->

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    for k in range(0, len(l1)):
        for z in psymptoms:
            if (z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = svclassifier.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if (predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        t4.delete("1.0", END)
        t4.insert(END, disease[a])
    else:
        t4.delete("1.0", END)
        t4.insert(END, "Not Found")



# gui_stuff------------------------------------------------------------------------------------

root = Tk()
root.configure(background='black')

# entry variables---------------------------------->

Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

# Heading---------------------------------------------->

w2 = Label(root, justify=LEFT, text="Disease Predictor", fg="black", bg="red")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=1, columnspan=2, padx=100)
w2.config(font=("Aharoni", 30))
w2.grid(row=2, column=1, columnspan=2, padx=100)

# labels------------------------------------------------>

NameLb = Label(root, text="Name of the Patient", fg="yellow", bg="black")
NameLb.grid(row=6, column=0, pady=15, sticky=W)


S1Lb = Label(root, text="Symptom 1", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="yellow", bg="black")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="yellow", bg="black")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="yellow", bg="black")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="yellow", bg="black")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

#------------------------------------------------------------>

lrLb = Label(root, text="RandomForest", fg="white", bg="red")
lrLb.grid(row=15, column=0, pady=10,sticky=W)

destreeLb = Label(root, text="ExtraTree", fg="white", bg="red")
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="NaiveBayes", fg="white", bg="red")
ranfLb.grid(row=19, column=0, pady=10, sticky=W)

svmLb = Label(root, text="SVM", fg="white", bg="red");
svmLb.grid(row=21,column=0, pady=10, sticky=W)

# entries----------------------------------------------------->

OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

#Symptoms----------------------------------------------------->

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)

#Algorithm Buttons---------------------------------------------->

dst = Button(root, text="RandomForest", command=RandomForest,bg="green",fg="yellow")
dst.grid(row=8, column=3,padx=10)

rnf = Button(root, text="ExtraTree", command=extraTree,bg="green",fg="yellow")
rnf.grid(row=9, column=3,padx=10)

lr = Button(root, text="NaiveBayes", command=NaiveBayes,bg="green",fg="yellow")
lr.grid(row=10, column=3,padx=10)

svm = Button(root , text="SVM",command=SVMFun, bg="green", fg="yellow")
svm.grid(row=11, column=3,padx=10)

#Output Fields--------------------------------------------------->

t1 = Text(root, height=1, width=40,bg="yellow",fg="black")
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40,bg="yellow",fg="black")
t2.grid(row=17, column=1 , padx=10)

t3 = Text(root, height=1, width=40,bg="yellow",fg="black")
t3.grid(row=19, column=1 , padx=10)

t4= Text(root , height=1, width=40, bg="yellow", fg="black")
t4.grid(row=21,column=1, padx=10)

root.mainloop()
