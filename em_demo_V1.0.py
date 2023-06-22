import numpy as np
import scipy
from scipy.optimize import fmin_slsqp
from scipy import integrate
from scipy import stats
from collections import Counter
from pdb import set_trace as st

def generate_2pl(nitem = 20, nperson = 40):
    beta  = np.random.randn(nitem)
    alpha  = np.exp(np.random.randn(nitem) / 2 )+ 0.7
    theta  = np.random.randn(nperson)
    return beta, alpha, theta


class TwoPlIRT:
    def __init__(self, person, item, response) -> None:
        self.person = person
        self.item = item
        self.response = response
        self.nperson = len(set(person))
        self.nitem = len(set(item))
        self.n_quadrature = 80
        self.quadrature_points = np.linspace(-4, 4, num = self.n_quadrature + 1)
        self.overall_quadrature_interval = [(self.quadrature_points[i], self.quadrature_points[i+1]) for i in range(self.n_quadrature)]
        self.quadrature_points_mean = np.array([np.mean([e[0], e[1]]) for e in self.overall_quadrature_interval])
        self.attemp_counter = np.array(list(Counter(item).values()))
        self.correct_counter = np.array([np.sum(response[self.item == i]) for i in range(self.nitem)])

    def _init_param(self):
        self.theta = np.random.randn(self.nperson)
        self.beta = np.random.randn(self.nitem)
        self.alpha = np.ones(self.nitem) + np.random.randn(self.nitem)

        # self.theta = np.zeros(self.nperson)
        # self.beta = np.zeros(self.nitem)
        # self.alpha = np.ones(self.nitem) 

    def irf(self, theta, alpha,beta):
        return 1 / (1+np.exp(beta- alpha * theta))

    def _estep(self):
        expected_attemp = np.zeros((len(self.beta), self.n_quadrature))
        expected_correct = np.zeros((len(self.beta), self.n_quadrature))
        
        population_theta_distribution = []
        
        for a,b in self.overall_quadrature_interval:
            _n = integrate.quad(func = stats.norm(0,1).pdf, a =a, b =b)
            population_theta_distribution.append(_n[0])

        for i in range(expected_attemp.shape[0]):
            expected_attemp[i,:] = population_theta_distribution
        expected_attemp *= self.attemp_counter.reshape(-1,1)
        
        for i in range(self.nitem):
            cur_alpha = self.alpha[i]
            cur_beta = self.beta[i]
            for j,(a,b) in enumerate(self.overall_quadrature_interval):
                _r = integrate.quad(func = lambda x: self.irf(x, cur_alpha, cur_beta), a=a, b=b)[0] / (b-a) #expect_r_ratio
                expected_correct[i, j] = _r
            expected_correct[i,:] *= expected_attemp[i,:] #expect_r
            expected_correct[i,:] *= self.correct_counter[i] / expected_correct[i,:].sum()
        return expected_correct, expected_attemp

    def _mstep(self, expected_correct, expected_attemp):
        """
        expected_attemp: shape like [n_item, n_quadrature], num of attemp in certain ability level
        expected_correct: shape like [n_item, n_quadrature], num of correct in certain ability level
        """
        
        for i in range(self.nitem):
            _expected_attemp = expected_attemp[i,:]
            _expected_correct = expected_correct[i,:]

            def objective(est):
                alpha, beta = est[0],est[1]
                loss =  np.log(1 / (1+np.exp(beta- alpha * self.quadrature_points_mean))) * _expected_correct + \
                    np.log(1 - 1 / (1+np.exp(beta- alpha * self.quadrature_points_mean))) * (_expected_attemp - _expected_correct)
                return -loss.sum()

            # print(f"--- start M step for {i} item: alpha {round(self.alpha[i],4)}, beta {round(self.beta[i],4)} ")
            otpt = fmin_slsqp(objective, (self.alpha[i], self.beta[i]),bounds=[(0.25, 4), (-6, 6)], disp = False)
            # otpt = fmin_slsqp(objective, (self.alpha[i], self.beta[i]), bounds=[(0.25, 4), (-6, 6)], full_output = True, iprint = 3)[0]
            self.alpha[i], self.beta[i] = otpt[0], otpt[1]
            # print(f"--- end M step for {i} item: alpha {round(self.alpha[i],4)}, beta {round(self.beta[i],4)} ")
        
nitem = 100
nperson = 100
beta, alpha, theta = generate_2pl(nitem = nitem, nperson = nperson)
person = np.repeat(list(range(nperson)), nitem) # n_user 20, n_question 30
item = np.tile(list(range(nitem)), nperson)
response = np.random.binomial(1, p = 1/(1+np.exp(beta[item] - alpha[item] *  theta[person])))

model = TwoPlIRT(person, item, response)
model._init_param()
for i in range(1000):
    expected_correct, expected_attemp = model._estep()
    model._mstep(expected_correct, expected_attemp)
    print(f"-- iter {i}--")
    print("a ", np.power(model.alpha - alpha, 2).sum())
    print("b ", np.power(model.beta - beta, 2).sum())
    print("ture_alpha: ", alpha[:5].round(3))
    print("model.alpha: ", model.alpha[:5].round(3))
    print("ture_beta: ", beta[:5].round(3))
    print("model.beta: ", model.beta[:5].round(3))