setwd("~/Scrivania")
data <- read.csv(file = "shockWL.csv", sep= ";")

par(mfrow=(c(2,3)))
x <- data[data$neig == 1,]
plot(x$ncomp,x$meanIso, type = "l", col = "slateblue1", lwd = 5, xlab = "n_components", ylab="Avg score", main="Isomap 1 neighbor")

x <- data[data$neig == 4,]
plot(x$ncomp,x$meanIso, type = "l", col = "slateblue1", lwd = 5, xlab = "n_components", ylab="Avg score", main="Isomap 4 neighbor")


x <- data[data$neig == 10,]
plot(x$ncomp,x$meanIso, type = "l", col = "slateblue1", lwd = 5, xlab = "n_components", ylab="Avg score", main="Isomap 10 neighbor")


x <- data[data$neig == 13,]
plot(x$ncomp,x$meanIso, type = "l", col = "slateblue1", lwd = 5, xlab = "n_components", ylab="Avg score", main="Isomap 13 neighbor")


x <- data[data$neig == 16,]
plot(x$ncomp,x$meanIso, type = "l", col = "slateblue1", lwd = 5, xlab = "n_components", ylab="Avg score", main="Isomap 16 neighbor")


x <- data[data$neig == 19,]
plot(x$ncomp,x$meanIso, type = "l", col = "slateblue1", lwd = 5, xlab = "n_components", ylab="Avg score", main="Isomap 19 neighbor")

par(mfrow=(c(1,1)))




par(mfrow=(c(2,3)))
x <- data[data$neig == 1,]
plot(x$ncomp,x$meanLLE, type = "l", col = "brown1", lwd = 5, xlab = "n_components", ylab="Avg score", main="LLE 1 neighbor")

x <- data[data$neig == 4,]
plot(x$ncomp,x$meanLLE, type = "l", col = "brown1", lwd = 5, xlab = "n_components", ylab="Avg score", main="LLE 4 neighbor")


x <- data[data$neig == 10,]
plot(x$ncomp,x$meanLLE, type = "l", col = "brown1", lwd = 5, xlab = "n_components", ylab="Avg score", main="LLE 10 neighbor")


x <- data[data$neig == 13,]
plot(x$ncomp,x$meanLLE, type = "l", col = "brown1", lwd = 5, xlab = "n_components", ylab="Avg score", main="LLE 13 neighbor")


x <- data[data$neig == 16,]
plot(x$ncomp,x$meanLLE, type = "l", col = "brown1", lwd = 5, xlab = "n_components", ylab="Avg score", main="LLE 16 neighbor")


x <- data[data$neig == 19,]
plot(x$ncomp,x$meanLLE, type = "l", col = "brown1", lwd = 5, xlab = "n_components", ylab="Avg score", main="LLE 19 neighbor")

par(mfrow=(c(1,1)))




