import numpy as np
from scipy.optimize import fmin_slsqp
from scipy import integrate
from scipy import stats
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

        self.person_item = [[] for i in range(self.nperson)]
        self.person_response = [[] for i in range(self.nperson)]
        for p, i, r in zip(self.person, self.item, self.response):
            self.person_item[p].append(i)
            self.person_response[p].append(r)
        self.person_item = [np.array(i) for i in self.person_item ]
        self.person_response = [np.array(i) for i in self.person_response ]

        self.item_person = [[] for i in range(self.nitem)]
        self.item_response = [[] for i in range(self.nitem)]
        for p, i, r in zip(self.person, self.item, self.response):
            self.item_person[i].append(p)
            self.item_response[i].append(r)

        self.item_person = [np.array(i) for i in self.item_person ]
        self.item_response = [np.array(i) for i in self.item_response ]
        self.n_quadrature = 80
        self.quadrature_points = np.linspace(-4, 4, num = self.n_quadrature + 1)
        self.running_loss = 0

    def _init_param(self):
        # self.theta = np.random.randn(self.nperson)
        # self.beta = np.random.randn(self.nitem)
        # self.alpha = np.ones(self.nitem) + np.random.randn(self.nitem)

        self.theta = np.zeros(self.nperson)
        self.beta = np.zeros(self.nitem)
        self.alpha = np.ones(self.nitem) 

    def _irf(self, theta, alpha, beta):
        return 1 / (1+np.exp(beta- alpha * theta))

    def _likehood_person(self, theta, alpha, beta, response):
        """
        theta: scaler
        alpha, beta, response: 1D numpy array"""
        c = 1 / (1+np.exp(beta- alpha * theta)) * response * 2
        w = (1 - 1 / (1+np.exp(beta- alpha * theta))) * (1 - response) * 2
        return np.prod(np.where(c == 0, 1, c)) * np.prod(np.where(w == 0, 1, w)) 

    def eap(self, prior_pdf, likelihood_func):
        eval_point, prior_p = prior_pdf[0], prior_pdf[1]
        posterior_p = [prob * likelihood_func(_eval_point) for _eval_point, prob in zip(eval_point, prior_p)]
        posterior_p_norm = np.array(posterior_p).sum()
        _eap = ((posterior_p / posterior_p_norm) * eval_point).sum()
        return _eap 

    def _estep(self):
        print("Estep...")
        prior_pdf = [np.linspace(-4,4, num = 81), [stats.norm(0,1).pdf(i) for i in np.linspace(-4,4, num = 81)]]
        for i in range(self.nperson):
            alpha = self.alpha[self.person_item[i]]
            beta = self.beta[self.person_item[i]]
            response = self.person_response[i]
            _likelihood_func = lambda x:self._likehood_person(x, alpha, beta, response)
            _eap = self.eap(prior_pdf, _likelihood_func)
            self.theta[i] = _eap

        
        self.beta = self.beta  - np.mean(self.theta)
        self.theta = self.theta  - np.mean(self.theta)

    def _mstep(self):
        print("Mstep...")
        self.running_loss = 0
        for i in range(self.nitem):
            def objective(est):
                alpha, beta = est[0],est[1]
                loss =  np.log(1 / (1+np.exp(beta- alpha * self.theta[self.item_person[i]]))) * self.item_response[i]  + \
                    np.log(1 - 1 / (1+np.exp(beta- alpha * self.theta[self.item_person[i]]))) * (1 - self.item_response[i])
                return -loss.sum()

            otpt = fmin_slsqp(objective, (self.alpha[i], self.beta[i]),bounds=[(0.25, 4), (-4, 4)], disp = False)
            self.alpha[i], self.beta[i] = otpt[0], otpt[1]
            self.running_loss += objective(otpt)
        

    def negloglikihood(self):
        _loglikihood =  np.sum(np.log(1/ (1+np.exp(self.beta[self.item] - self.alpha[self.item] * self.theta[self.person]))) * self.response) + \
             np.sum(np.log(1 - 1/ (1+np.exp(self.beta[self.item] - self.alpha[self.item] * self.theta[self.person]))) * (1 - self.response))
        _negloglikihood = - _loglikihood
        print("_negloglikihood: ", _negloglikihood / len(self.response))
        return _negloglikihood / len(self.response)

nitem = 1000
nperson = 1000
beta, alpha, theta = generate_2pl(nitem = nitem, nperson = nperson)
person = np.repeat(list(range(nperson)), nitem) 
item = np.tile(list(range(nitem)), nperson)
response = np.random.binomial(1, p = 1/(1+np.exp(beta[item] - alpha[item] *  theta[person])))

model = TwoPlIRT(person, item, response)
model._init_param()
negloglikihood_pre = float("inf")
for i in range(200):
    print(f"--- iteration {i} ---")
    model._estep()
    model._mstep()
    print("loss", model.running_loss / len(response))
    # negloglikihood = model.negloglikihood()

    print("alpha_abs_error", np.abs(model.alpha - alpha).mean())
    print("beta_abs_error", np.abs(model.beta - beta).mean())
    print("theta_abs_error ", np.abs(model.theta - theta).mean())

    print("alpha_bia", (model.alpha - alpha).mean())
    print("beta_bia", (model.beta - beta).mean())
    print("theta_bia ", (model.theta - theta).mean())

    print("ture_alpha: ", alpha[:5].round(3))
    print("model.alpha: ", model.alpha[:5].round(3))
    print("ture_beta: ", beta[:5].round(3))
    print("model.beta: ", model.beta[:5].round(3))
    print("ture_theta: ", theta[:5].round(3))
    print("model.theta: ", model.theta[:5].round(3))
