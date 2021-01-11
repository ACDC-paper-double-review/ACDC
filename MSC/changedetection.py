from properties import Properties
import math, numpy as np
from scipy.stats import beta, binom
from decimal import Decimal
import sys, random, time


class ChangeDetection(object):

	def __init__(self, gamma, sensitivity, maxWindowSize):
		self.gamma = gamma
		self.sensitivity = sensitivity
		self.maxWindowSize = maxWindowSize


	"""
	 Functions to estimate beta distribution parameters
	"""
	def __calcBetaDistAlpha(self, list, sampleMean, sampleVar):
		if sampleMean == -1:
			sampleMean = np.mean(list)
		if sampleVar == -1:
			sampleVar = np.var(list)
		c = (sampleMean * (1-sampleMean)/sampleVar) - 1
		return sampleMean * c


	def __calcBetaDistBeta(self, list, alphaChange, sampleMean):
		if sampleMean == -1:
			sampleMean = np.mean(list)
		return alphaChange * ((1.0/sampleMean) - 1)


	"""
	input: The dynamic sliding window containing confidence of target classifier
    output: -1 if no change found, otherwise the change point
	"""
	def detectTargetChange(self, slidingWindow):
		estimatedChangePoint = -1
		N = len(slidingWindow)
		cushion = max(Properties.CUSHION, int(math.floor(N ** self.gamma)))

		#If mean confidence fall below 0.3, must retrain the classifier, so return a changepoint
		if N > self.maxWindowSize:
			Properties.logger.info('Current target Window Size is: ' + str(N) + ', which exceeds max limit, so update classifier')
			return 0
		if N > 2*cushion and np.mean(slidingWindow[0:N]) <= Properties.CONFCUTOFF:
			Properties.logger.info('Current target Window Size is: ' + str(N))
			Properties.logger.info('But overall confidence fell below ' + str(Properties.CONFCUTOFF) + ', so update classifier')
			return 0

		threshold = -math.log(self.sensitivity)
		w = 0.0
		kAtMaxW = -1

		kindex = np.arange(cushion, N - cushion + 1)
		for k in kindex:
			xbar0 = np.mean(slidingWindow[:k])
			var0 = np.var(slidingWindow[:k])
			xbar1 = np.mean(slidingWindow[k:])
			var1 = np.mean(slidingWindow[k:])

			if xbar1 <= 0.9*xbar0:
				skn = 0.0
				alphaPreChange = self.__calcBetaDistAlpha(slidingWindow[:k], xbar0, var0)
				betaPreChange = self.__calcBetaDistBeta(slidingWindow[:k], alphaPreChange, xbar0)
				alphaPostChange = self.__calcBetaDistAlpha(slidingWindow[k:], xbar1, var1)
				betaPostChange = self.__calcBetaDistBeta(slidingWindow[k:], alphaPostChange, xbar1)

				# for i in range(k, N):
				# 	try:
				# 		#Get Beta distribution
				# 		denom = beta.pdf(float(slidingWindow[i]), alphaPreChange, betaPreChange)
				# 		if denom == 0:
				# 			Properties.logger.info('beta distribution pdf is zero for X = ' + str(float(slidingWindow[i])))
				# 			denom = 0.001
				# 		val = Decimal(beta.pdf(float(slidingWindow[i])/denom, alphaPostChange, betaPostChange))
				# 		skn += float(val.ln())
				# 	except:
				# 		e = sys.exc_info()
				# 		print str(e[1])
				# 		raise Exception('Error in calculating skn.')

				try:
					swin = map(float, slidingWindow[k:])
					denom = [beta.pdf(s, alphaPreChange, betaPreChange) for s in swin]
					nor_denom = np.array([0.001 if h == 0 else h for h in denom])
					nor_swin = swin/nor_denom
					skn = sum([Decimal(beta.pdf(ns, alphaPostChange, betaPostChange)).ln() for ns in nor_swin])
				except:
					e = sys.exc_info()
					print str(e[1])
					raise Exception('Error in calculating skn')

				if skn > w:
					w = skn
					kAtMaxW = k

		if w >= threshold and kAtMaxW != -1:
			estimatedChangePoint = kAtMaxW
			Properties.logger.info('Estimated change point is ' + str(estimatedChangePoint) + ', detected at ' + str(N))


		return estimatedChangePoint



	"""
	input: The dynamic sliding window containing accuracy of source classifier
    output: -1 if no change found, otherwise the change point
	"""
	def detectSourceChange(self, slidingWindow):
		estimatedChangePoint = -1
		N = len(slidingWindow)
		cushion = max(Properties.CUSHION, int(math.floor(N ** self.gamma)))

		#If mean confidence fall below 0.3, must retrain the classifier, so return a changepoint
		if N > self.maxWindowSize:
			Properties.logger.info('Current target Window Size is: ' + str(N) + ', which exceeds max limit, so update classifier')
			return 0
		if N > 2*cushion and np.mean(slidingWindow) <= Properties.CONFCUTOFF:
			Properties.logger.info('Current target Window Size is: ' + str(N))
			Properties.logger.info('But overall confidence fell below ' + str(Properties.CONFCUTOFF) + ', so update classifier')
			return 0

		threshold = -math.log(self.sensitivity)
		w = 0.0
		kAtMaxW = -1

		kindex = np.arange(cushion, N - cushion + 1)
		for k in kindex:
			xbar0 = np.mean(slidingWindow[:k])
			xbar1 = np.mean(slidingWindow[k:])

			# means should set 1=accurate, 0=erroneous
			if xbar1 <= 0.9*xbar0:
				skn = 0.0

				# for i in range(k, N):
				# 	try:
				# 		denom = binom.pmf(float(slidingWindow[i]), k, xbar0)
				# 		if denom == 0:
				# 			Properties.logger.info('binomial distribution pmf is zero for X = ' + str(float(slidingWindow[i])))
				# 			denom = 0.001
				# 		val = Decimal(binom.pmf(float(slidingWindow[i])/denom, N-k, xbar1))
				# 		skn += float(val.ln())
				# 	except:
				# 		e = sys.exc_info()
				# 		print str(e[1])
				# 		raise Exception('Error in calculating skn')

				try:
					swin = map(float, slidingWindow[k:N])
					denom = [binom.pmf(s, k, xbar0) for s in swin]
					nor_denom = np.array([0.001 if h == 0 else h for h in denom])
					nor_swin = swin/nor_denom
					skn = sum([Decimal(binom.pmf(ns, N-k, xbar1)).ln() for ns in nor_swin])
				except:
					e = sys.exc_info()
					print str(e[1])
					raise Exception('Error in calculating skn')

				if skn > w:
					w = skn
					kAtMaxW = k

		if w >= threshold and kAtMaxW != -1:
			estimatedChangePoint = kAtMaxW
			Properties.logger.info('Estimated change point is ' + str(estimatedChangePoint) + ', detected at: ' + str(N))
			Properties.logger.info('Value of w: ' + str(w) + ', Value of Threshold: ' + str(threshold))


		return estimatedChangePoint
