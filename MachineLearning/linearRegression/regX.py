#I am a calculator for linear regression
#I want a panda dataframe with the important columns for the regression with the name "Age" and "charges"
#xk = "Age" and yk = "charges"
import pandas as pd

class regX:
            
    def calc(df):
        #adding panda.core.series.Series for calculation of the mean
        Meanbyage = df.groupby('age').charges.mean()
        
        #converting to panda.Dataframe
        MeanByAge = Meanbyage.to_frame()
        
        #The required sums:    
        #u = Sum of all (xk **2)
        #v = Sum of all xk
        #w = Sum of all yk
        #z = Sum of all (xk * yk)
        
        #preparing the dataframe for the claculation
        MeanByAge['Age'] = MeanByAge.index
        MeanByAge['u'] = MeanByAge.Age **2
        MeanByAge['z'] = MeanByAge.Age * MeanByAge.charges
        
        #calculation of the sums
        u = MeanByAge['u'].sum()
        v = MeanByAge['Age'].sum()
        w = MeanByAge['charges'].sum()
        z = MeanByAge['z'].sum()
        n = len(MeanByAge)
        
        #lineare regression
        m = round((n*z -v*w)/(n*u - v**2),20)
        b = round((u*w-v*z)/(n*u -v**2),10)
        list = [round(m,3),round(b,3)]
        
        return list