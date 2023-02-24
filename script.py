#import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import codecademylib3

#load data
forests = pd.read_csv('forests.csv')

#check multicollinearity with a heatmap

corr_grid = forests.corr()
sns.heatmap(corr_grid, xticklabels=corr_grid.columns, yticklabels=corr_grid.columns, annot=True)


#plot humidity vs temperature
sns.lmplot(x='temp',y='humid',hue='region',data=forests, fit_reg = False)



#plot humidity vs temperature


# Add regression line for model0 here:
modelH = sm.OLS.from_formula('humid ~ temp + region', data=forests).fit()
print(modelH.params)
# Print coefficients here:

#model predicting humidity
plt.plot(forests.temp, modelH.params[0]+modelH.params[1]*0+modelH.params[2]*forests.temp, color='blue',linewidth=5, label='Bejaia')
plt.plot(forests.temp, modelH.params[0]+modelH.params[1]*1+modelH.params[2]*forests.temp, color='blue',linewidth=5, label='Sid Bel-abbes')

#equations


#interpretations
## Coefficient on temp:

## For Bejaia equation:

## For Sidi Bel-abbes equation:


#plot regression lines


#plot FFMC vs temperature
sns.lmplot(x='temp',y='FFMC',data=forests, fit_reg = False)
plt.show()
plt.clf()


#model predicting FFMC with interaction
resultsF = sm.OLS.from_formula('FFMC ~ temp + fire+temp:fire', data=forests).fit()
print(resultsF.params)
#equations
## Full equation:


plt.plot(forests.temp, resultsF.params[0]+resultsF.params[1]*0+resultsF.params[2]*forests.temp + resultsF.params[3]*forests.temp*0, color='blue',linewidth=5, label='No Fire')
## For locations without fire:

## For locations with fire:
plt.plot(forests.temp, resultsF.params[0]+resultsF.params[1]*1+resultsF.params[2]*forests.temp + resultsF.params[3]*forests.temp*1, color='blue',linewidth=5, label='Fire')

#interpretations
## For locations without fire:

## For locations with fire:


#plot regression lines
sns.lmplot(x='temp',y='FFMC',hue='fire',data=forests, fit_reg = False)
plt.show()
plt.clf()

#plot FFMC vs humid
sns.lmplot(x='humid',y='FFMC',data=forests, fit_reg = False)
plt.show()
plt.clf()

#polynomial model predicting FFMC
resultsP = sm.OLS.from_formula('FFMC ~ humid + np.power(humid,2)', data=forests).fit()
print(resultsP.params)
modelstr = sm.OLS.from_formula('FFMC ~ humid',data=forests).fit()
print(modelstr.params)



plt.plot(forests.humid, modelstr.params[0]+modelstr.params[1]*forests.humid, color='blue',linewidth=5, label='Fire')
plt.show()
plt.clf()
#regression equation
print(resultsP.params[0] + resultsP.params[1]*25 + resultsP.params[2]*np.power(25,2))
print(resultsP.params[0] + resultsP.params[1]*35 + resultsP.params[2]*np.power(35,2))
print(resultsP.params[0] + resultsP.params[1]*60 + resultsP.params[2]*np.power(60,2))
print(resultsP.params[0] + resultsP.params[1]*70 + resultsP.params[2]*np.power(70,2))
#sample predicted values


#interpretation of relationship


#multiple variables to predict FFMC
sns.lmplot(x='BUI',y='FWI',hue='fire',data=forests, fit_reg = False)
plt.show()
plt.clf()
sns.lmplot(x='ISI',y='FWI',hue='fire',data=forests, fit_reg = False)
plt.show()
plt.clf()
#predict FWI from ISI and BUI
modelFFMC = sm.OLS.from_formula('FFMC~humid+temp+wind+rain',data=forests).fit()
print(modelFFMC.params)