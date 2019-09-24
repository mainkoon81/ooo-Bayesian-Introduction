#### Bayes Rule
<img src="https://user-images.githubusercontent.com/31917400/34920230-5115b6b6-f967-11e7-9493-5f6662f1ce70.JPG" width="400" height="500" />

We know the Bayes rule. How does it relate to machine learning? Bayesian inference is based on using probability to represent **all forms of uncertainty**.

## Introduction
 - Frequentists' probability that doesn’t depend on one’s beliefs refers to **past events**..Do experiment and that's it.   
 - Bayesians' probability as a measure of beliefs refers to **future events**..posterior 

As Bayesians, we start with a belief, called a prior. Then we obtain some data and use it to update our belief. The outcome is called a posterior. Should we obtain even more data, the old posterior becomes a new prior and the cycle repeats.
<img src="https://user-images.githubusercontent.com/31917400/64063569-a9533100-cbed-11e9-89d6-a8cc6203b886.jpg"/>

`P( θ | Data ) = P( Data | θ ) * P( θ ) / P( data )`
 - `P( θ )` is a prior, our belief of what the model parameters might be. 
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
         - Taking a **center* and **spread** as arguments, it states that 67% of your data is within 1*SD of the center, and 95% is within 2*SD. 
           - No need to check our value boundaries. 
         - coming up a lot because if you have multiple signals that come from any distribution, with enough signals their average converges to the normal distribution. `hist(np.array([np.mean(your_distribution) for i in range(your_samples)]))`.
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
   
 - `P( θ | Data )`, a posterior, is what we’re after. 
   - It’s a parametrized distribution over model parameters obtained from prior beliefs and data.

 - `P( Data | θ )` is called likelihood of data given model parameters. 
   <img src="https://user-images.githubusercontent.com/31917400/65486881-8c80e500-de9d-11e9-9d6b-e8d7b8af1d09.jpg"/>
   - **The formula for likelihood is model-specific**. 
   - People often use likelihood for evaluation of models: a model that gives higher likelihood to real data is better.
   - When one uses likelihood to get point estimates of model parameters, it’s called  MLE. 
   - If one also takes the prior into account, then it’s maximum a posteriori estimation (MAP). 
   - MLE and MAP are the same if the prior is uniform of course.






























































































