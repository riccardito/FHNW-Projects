import pandas as pd
import matplotlib.pyplot as plt
import regX

#Der einzulesende Text, als exel
df = pd.read_excel('insurance.xlsx')

result= regX.regX.calc(df)
print("Funktion = m:",result[0],"und b:",result[1])
