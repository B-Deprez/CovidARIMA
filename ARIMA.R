library(COVID19)
library(forecast)
library(ggplot2)
library(tidyverse)

df <- covid19(country = 56, level = 1, start = "2020-03-01", end =
          Sys.Date()-5, raw = TRUE, vintage = FALSE, verbose = TRUE, cache =
          TRUE, wb = NULL, gmr = NULL, amr = NULL) %>%
  mutate(new_cases = c(0, diff(confirmed)))

sda = c()
for(i in 1:length(df$date)){
  sda[i] <- mean(df$new_cases[max(1,i-6):i])
}

df$seven_day_average <- sda

ggplot(df, aes(date, new_cases)) +
  geom_line()+
  theme_bw()

ggplot(df, aes(date, seven_day_average))+
  geom_line()+
  theme_bw()

model_1 <- auto.arima(df$new_cases); model_1
acf(model_1$residuals)
plot(forecast(model_1, 31, bootstrap = T))


model_7 <- auto.arima(df$seven_day_average); model_7
acf(model_7$residuals)
plot(forecast(model_7, 31, bootstrap = T))

df <- df %>%
  mutate(seven_day_fit = fitted(model_7),
         new_cases_fit = fitted(model_1))

ggplot(df, aes(date, seven_day_average))+
  geom_line(color = "blue") +
  geom_line(aes(date,seven_day_fit), color = "red")+
  theme_bw()

ggplot(df, aes(date, new_cases))+
  geom_line(color = "blue") +
  geom_line(aes(date,new_cases_fit), color = "red")+
  theme_bw()

plot(diff(df$new_cases))

plot(diff(df$seven_day_average))
