import pandas as pd
import matplotlib.pyplot as plt
import regX

#Der einzulesende Text, als exel
df = pd.read_excel (r'C:\Studium\1.Semester_DS\Challenge Krankenkassenprämie\insurance.xlsx')

result= regX.regX.calc(df)
print("Funktion = m:",result[0],"und b:",result[1])
