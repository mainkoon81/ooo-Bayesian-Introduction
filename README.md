#### Bayes Rule
<img src="https://user-images.githubusercontent.com/31917400/34920230-5115b6b6-f967-11e7-9493-5f6662f1ce70.JPG" width="400" height="500" />

We know the Bayes rule. How does it relate to machine learning? Bayesian inference is based on using probability to represent **all forms of uncertainty**.

## [Uncertainty]
 - **Aleatory variability** is the natural(intrinsic) randomness in a process; it is supposed **irreducible** and inherent natural to the process involved. 
   - Heteroscedastic: No one can sure the measurements done by your collegues are perfect..damn noise...(heteroscedastic means a different uncertainty for every input)
   - Homoscedastic: model variance? you assumes identical observation noise for every input point x? Instead of having a variance being dependent on the input x, we must determine a so-called model precision **`τ`** and multiply it by the identity matrix I, such that all outputs y have the same variance and no co-variance among them exists. This model precision **`τ`** is the inverse observation standard deviation.
 - **Epistemic uncertainty** is the scientific uncertainty in the model of the process; it is supposedly **reducible** with better knowledge, since it is not inherent in the real-world process under consideration (due to lack of knowledge and limited data..This can be reduced in time, if more data are collected and new models are developed). 


# 1> Introduction
 - Frequentists' probability that doesn’t depend on one’s beliefs refers to **past events**..Do experiment and that's it.   
 - Bayesians' probability as a measure of beliefs refers to **future events**..posterior..Do update !  

As Bayesians, we start with a belief, called a prior. Then we obtain some data and use it to update our belief. The outcome is called a posterior. Should we obtain even more data, the old posterior becomes a new prior and the cycle repeats. It's very honest. We cannot 100% rely on the experiment result. There is always a discrepency and there is no guarantee that the relative frequency of an event will match the true underlying probability of the event. That’s why we are approximating the probability by the long-run relative frequency in Bayesian. It's like calibrating your frequentist's subjective belief.  
<img src="https://user-images.githubusercontent.com/31917400/66057262-52b07e00-e530-11e9-8a97-3eac1d67d76e.jpg"/>

`P( θ | Data ) = P( Data | θ ) * P( θ ) / P( data )`

### Prior
 - `P( θ )` is a prior, our belief of what the model parameters might be. 
   - Prior is a weigth or regularizor. 
   - Most often our opinion in this matter is rather vague and if we have enough data, we simply don’t care. 
   - Inference should converge to probable `θ` as long as it’s not zero in the prior. 
   - It's a parametrized distribution.
   - why a paricular prior was chosen? 
     - The reality is that many of these prior distributions are making assumptions about the **`type of data`** we have.
     - There are some distributions used again and again, but the others are special cases of these dozen or can be created through a clever combination of two or three of these simpler distributions.
     - A prior is employed because the assumptions of the prior match what we know about the **parameter generation process**.
     - *Actually, there are multiple effective priors for a particular problem. A particular prior is chosen as some combination of `analytic tractability` + `computationally efficiency`, which makes other recognizable distributions when combined with popular likelihood functions. 
     - Examplary Distributions
       - __Uniform distribution__
         - Whether you use this one in its continuous case or its discrete case, it is used for the same thing: 
           - `You have a set of events that are equally likely`.
         - Note, the uniform distribution from ∞ to −∞ is not a probability distribution. 
           - Need to give **lower** and **upper** bounds for our values.
           - Not used as often as you’d think, since its rare we want hard boundaries on our values.
       - __Gaussian distribution__
         - Taking a **center** and **spread** as arguments, it states that 67% of your data is within 1*SD of the center, and 95% is within 2*SD. 
           - No need to check our value boundaries. 
         - coming up a lot because if you have multiple signals that come from any distribution (with enough signals), their average always converges to the normal distribution. `hist(np.array([np.mean(your_distribution) for i in range(your_samples)]))`.
       - __Bernoulli distribution__
         - It is handy since you can define a bunch of distributions in terms of them???
         - The multinomial can be used to encode our beliefs about each vocabulary.
       - __Gamma distribution__
         - It comes up all over the place. The intuition for the gamma is that it is the prior on **positive real numbers**. 
           - Now there are many ways to get a distribution over positive numbers.
             - take the `absolute-value of a normal distribution` and get what’s called a **Half-Normal distribution**. 
             - take the `exp(Y)` and `Y^2`...**Log-Normal**, and **χ-square**. 
         - So why use the gamma prior? 
           - If you use a **Log-Normal**, you are implicitly saying that you expect the **log** of your variable is symmetric.
           - If you use a **χ-square**, you are implicitly saying that your variable is the **sum of k?-squared factors**, where each factor came from the normal(0, 1) distribution.
           - Some people suggest using gamma because it is conjugate with lots of distributions. so it makes performing a computation easier...but it would be better to have your priors actually encode what you believe. 
             -  When gamma is a used as the **prior** to something like normal, the **posterior** of this distribution also is a gamma. 
           - The gamma distribution is the main way to encode something to be a postive number. Actually many distributions can be built from gamma.
             - It’s parameters `shape`(k) and `scale`(θ) roughly let you tune gamma like the normal distribution. kθ specifies the mean, and kθ^2 specifies the variance. 
             - Taking the reciprocal of a variable from the gamma gives you a value from the Inv-gamma distribution. 
             - If we normalize this positive number, we get the Beta distribution.
             ```
             def beta(a, b):
                 def samples(s):
                     x = r.gamma(a, 1, s)
                     y = r.gamma(b, 1, s)
                     return x/(x + y)
                 return(samples)
             ```
             - If we want to a prior on "categorical", which takes as an argument a list of numbers that sum to 1, we can use a gamma to generate k-numbers and then normalize. This is precisely the definition of the Dirichlet distribution. 
       - __Heavy-tailed distribution__
         - The major advantage of using a heavy-tail distribution is it’s more robust towards outliers (we cannot be too optimistic about how close a value stays near the mean..)..let's start to care outliers..
         - `t-distribution` can be interpretted as the distribution over a **sub-sampled population** from the normal distribution sample. Since here our sample size is so small, atypical values can occur more often than they do in the general population. As our sub-population grows, the t-distribution becomes the normal distribution. 
           - The t-distribution can also be generalized to not be centered at 0.
           - The parameter `ν` lets you state how large you believe this subpopulation to be.
         - `Laplace-distribution` as an interesting modification to the normal distribution(replacing `exp(L2-norm)` with `exp(L1-norm)` in the formula). A laplace centered on 0 can be used to put a strong **sparsity prior** on a variable while leaving a heavy-tail for it if the value has strong support for another value. 
   
### Posterior: Inference(Parameter Estimation in Bayesian way)
 - `P( θ | Data )`, a posterior, is what we’re after. 
   - It’s a parametrized distribution over model parameters obtained from prior beliefs and data.
   - In Bayesian Inference, "Inference" refers to how you learn parameters of your model. There are two main flavours:   
     - **1. Inference using Monte Carlo sampling:** a gold standard, but slow. 
     - **2. Variational inference:** It is designed explicitly to trade some accuracy for speed. It’s drawback is that it’s model-specific, but there’s light at the end of the tunnel...  

### Maximum Likelihood(Parameter Estimation with Frequentist's pride)
 - `P( Data | θ )` is called likelihood of data given model parameters. 
 <img src="https://user-images.githubusercontent.com/31917400/65486881-8c80e500-de9d-11e9-9d6b-e8d7b8af1d09.jpg"/>
  
 - **The formula for likelihood is model-specific**. 
   - People often use likelihood for evaluation of models: a model that gives higher likelihood to real data is better.
   - When one uses likelihood to get point estimates of model parameters, it’s called  MLE. 
   - If one also takes the prior into account, then it’s maximum a posteriori estimation (MAP). 
   - MLE and MAP are the same if the **prior is uniform**.

### Evidence: Prediction(Data value Prediction)
Let's train data points X and Y. We want predict the new Y at the end. In Bayesian Prediction, the predicted value is a **weighted average** of output of our model for all possible values of parameters. 
<img src="https://user-images.githubusercontent.com/31917400/66065180-c0fc3d00-e53e-11e9-89ed-2dc98835b11b.jpg"/>

### Modeling
 - The Bayesian Network!! (Probabilistic Graphical Model?????)
 
In the settings where data is scarce and precious and hard to obtain, it is difficult to conduct a large-scale controlled experiment, thus we cannot spare any effort to make the best use of available input. `With small data, it is important to **quantify uncertainty**` and that’s precisely what Bayesian approach is good at. In Bayesian Modeling, there are two main flavours:
 - **1. Statistical Modeling:** 
   - Multilevel/Hierarchical Modeling(Regression?)
 - **2. probabilistic ML including non-parametric approaches:** using data for a computer to learn automatically from it.
   - it outputs probabilistic predictions...that's why probabilistic.. also these probabilities are only statements of belief from a classifier.
   - __Generative modeling:__ One can sample or generate examples from it. Compare with classifiers(discriminative model to model `P(y|x)` to discriminate between classes based on x), **a generative model is concerned with joint distribution `P(y,x)`**. It’s more difficult to estimate that distribution, but **it allows sampling** and of course one can get `P(y|x)` from `P(y,x)`.
     - **LDA:** You start with a matrix where `rows` are **documents**, `columns` are **words** and `each element` is a **count of a given word** in a given document. LDA “factorizes” this matrix of size n x d into two matrices, documents/topics (n x k) and topics/words (k x d). you can’t multiply those two matrices to get the original, but since the appropriate rows/columns sum to one, **you can “generate” a document**. 
   - __Bayesian non-parametrics Modeling:__ the number of parameters in a model can grow as more data become available. This is similar to SVM, for example, where the algorithm chooses support vectors from the training points. Nonparametrics include **Hierarchical Dirichlet Process** version of LDA(where the number of topics chooses itself automatically), and **Gaussian Processes**.
     - **Gaussian Processes:** It is somewhat similar to SVM - both use **kernels** and have similar **scalability**(which has been vastly improved throughout the years by using approximations). 
       - A natural formulation for GP is **`regression`**, with **classification** as an afterthought. For SVM, it’s the other way around. As most “normal” methods provide **point estimates**, Bayesian counterparts, like Gaussian processes, also output **uncertainty estimates** while SVM are not. Even a sophisticated method like GP normally operates on an **assumption of homoscedasticity**, that is, uniform noise levels. In reality, noise might differ across input space (be heteroscedastic). 
       - Gaussian Distribution is ...



# 1. Knowledge



























































































