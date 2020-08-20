library(ggplot2)
library(MASS)

set.seed(1234)
meat_mu <- c(175, 90)
veg_mu <- c(175, 85)

sd <- c(15,20)
cor <- matrix(c(1,0.7,0.7,1), ncol = 2)

cov <- diag(sd) %*% cor %*% diag(sd)

n <- 500
# n <- 25
meat <- mvrnorm(n, meat_mu, cov)
veg <- mvrnorm(n, veg_mu, cov)

df <- rbind(meat, veg)
df <- data.frame(df)
df$diet <- sort(rep(c("meat", "veg"), n))
names(df) <- c("height", "weight", "diet")
df$age <- round(rnorm((n*2),45,20),0)
head(df)

write.csv(df, "/Users/chelseaparlett/Desktop/heightWeightBIG.csv")
ggplot(df, aes(height, weight)) + geom_point() + geom_smooth(method = "lm", se = F, aes(color = diet)) +
  theme_minimal()
summary(lm(weight ~ height + diet + age, data = df))

dfS <- df
dfS$height <- scale(df$height)
dfS$age  <- scale(df$age)

summary(lm(weight ~ height + diet + age, data = dfS))
