import numpy as np

class PID():
	def __init__(self,P,I,D,PVindex,SVindex,scanrate=1):
		self.P = P
		self.I = I
		self.D = D
		self.PVindex = PVindex
		self.SVindex = SVindex
		self.scanrate = scanrate
	
	def step (self,state):
		'''
		PVDeltaErrorPrevious = tag.PVDeltaError
		PVDeltaError = PVError - tag.PVPreviousError
		tag.PVDeltaError = PVDeltaError

		Kp = (100.0 / tag.P) * ((tag.MSH - tag.MSL) / (tag.SH - tag.SL))

		Ki = PVError * (_ScanTime / tag.I)

		Kd = (tag.D / _ScanTime) * PVDeltaErrorPrevious

		MVDelta = Kp * (PVDeltaError + Ki + Kd)

		tag.MV += MVDelta
		'''
		PrevPV = state[state.shape[0]-2,self.PVindex]
		PrevSV = SV = state[state.shape[0]-2,self.SVindex]
		PV = state[state.shape[0]-1,self.PVindex]
		SV = state[state.shape[0]-1,self.SVindex]
		
		Prev_error = PrevSV-PrevPV
		error = SV-PV
		
		PVDeltaError = error - Prev_error

		Kp = self.P
		Ki = error *(self.scanrate / self.I)
		Kd = (self.D / self.scanrate) * Prev_error

		MVDelta = Kp * (PVDeltaError + Ki + Kd)
		
		return MVDelta 

