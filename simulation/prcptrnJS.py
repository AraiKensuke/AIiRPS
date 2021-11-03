import numpy as _N


class perceptronJS:
    pred=None
    prc_weight=None
    prc_record=None
    ini_prc_weight=None
    fin_prc_weight=None

    def __init__(self, NN):
        self.pred = _N.zeros(3)
        self.prc_N = NN
        self.prc_weight = _N.random.rand(3, 3, self.prc_N)*4 - 2
        self.prc_record = _N.zeros((3, self.prc_N))

        self.ini_prc_weight=""
        for i in range(3):
            for j in range(3):
                for k in range(self.prc_N):
                    self.ini_prc_weight += "%.4f " % self.prc_weight[i][j][k]
	
    def predict(self, player):
        # current player move is predicted by current contents of this.pred
        prec =   _N.ones(3) * -1
        prec[player-1] = 1

        #print("...................In predict  player:   %d" % player)

        #print("old prc_record")
	#console.log(this.prc_record[0][0] + " " + this.prc_record[0][1] + " " + this.prc_record[1][0] + " " + this.prc_record[1][1] + " " + this.prc_record[2][0] + " " + this.prc_record[2][1])
	
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(self.prc_weight)

        for i in range(3):
            if prec[i]*self.pred[i] <= 0:
                for j in range(3):
                    for k in range(self.prc_N):
                        self.prc_weight[i,j,k] += prec[i]*self.prc_record[j,k]

        workspace = _N.empty(self.prc_N+1)
        #
        for i in range(3):
            workspace[1:self.prc_N+1]      = self.prc_record[i]
            workspace[0]        = prec[i]
            self.prc_record[i] = workspace[0:self.prc_N]

        # print("new prc_record")
        # print(self.prc_record[0])
        # print(self.prc_record[1])
        # print(self.prc_record[2])	

	#  Build new prediction of what HUMAN will play.
        self.pred[:] = 0

        for i in range(3):
            for j in range(3):
                for k in range(self.prc_N):
                    self.pred[i] += self.prc_weight[i,j,k]*self.prc_record[j,k]

        maxval=self.pred[0]
        maxnum = 0

        for i in range(3):
            if self.pred[i] >= maxval:
                maxval = self.pred[i]
                maxnum = i

        #print("****************************")
        #print(self.prc_weight)
        #outstr = self.prc_weight[0, 0] + "\n" +	self.prc_weight[0][1] + "\n" +	self.prc_weight[0][2] + "\n\n" +		self.prc_weight[1][0] +	"\n" +	self.prc_weight[1][1] + "\n" +		self.prc_weight[1][2] + "\n\n" +		self.prc_weight[2][0] + "\n" +		self.prc_weight[2][1] + "\n" +		self.prc_weight[2][2];
        #print(outstr)
	
        return maxnum+1

    def done(self):
        self.fin_prc_weight = ""
        for i in range(3):
            for j in range(3):
                for k in range(self.prc_N):
                    self.fin_prc_weight += "%.4f " % self.prc_weight[i][j][k]
