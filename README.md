#### Bayes Rule
<img src="https://user-images.githubusercontent.com/31917400/34920230-5115b6b6-f967-11e7-9493-5f6662f1ce70.JPG" width="400" height="500" />

We know the Bayes rule. How does it relate to machine learning? Bayesian inference is based on using probability to represent **all forms of uncertainty**.

## [Uncertainty]
 - **Aleatory variability** is the natural(intrinsic) randomness in a process; it is supposed **irreducible** and inherent natural to the process involved. 
   - Heteroscedastic: No one can sure the measurements done by your collegues are perfect..damn noise...(heteroscedastic means a different uncertainty for every input)
   - Homoscedastic: model variance? you assumes identical observation noise for every input point x? Instead of having a variance being dependent on the input x, we must determine a so-called model precision **`τ`** and multiply it by the identity matrix I, such that all outputs y have the same variance and no co-variance among them exists. This model precision **`τ`** is the inverse observation standard deviation.
 - **Epistemic uncertainty** is the scientific uncertainty in the model of the process; it is supposedly **reducible** with better knowledge, since it is not inherent in the real-world process under consideration (due to lack of knowledge and limited data..This can be reduced in time, if more data are collected and new models are developed). 

---------------------------------------------------------------------------------------------------------
# 1> Inference & Prediction
 - Inference for **θ** aims to understand the model.
 - Prediction for **Data** aims to utilize the model you discovered.
 - Frequentists' probability refers to **past events**..Do experiment and that's it.   
 - Bayesians' probability refers to **future events**..Do update !  

As Bayesians, we start with a belief, called a prior. Then we obtain some data and use it to update our belief. The outcome is called a posterior. Should we obtain even more data, the old posterior becomes a new prior and the cycle repeats. It's very honest. We cannot 100% rely on the experiment result. There is always a discrepency and there is no guarantee that the relative frequency of an event will match the true underlying probability of the event. That’s why we are approximating the probability by the long-run relative frequency in Bayesian. It's like calibrating your frequentist's subjective belief  
<img src="https://user-images.githubusercontent.com/31917400/66057262-52b07e00-e530-11e9-8a97-3eac1d67d76e.jpg"/>

`P( θ | Data ) = P( Data | θ ) * P( θ ) / P( data )`

## [a] `Prior`
 - `P( θ )` is a prior, our belief of what the model parameters might be. 
   - Prior is a weigth or regularizor. 
   - The final inference should converge to probable `θ` as long as it’s not zero in the prior.
   - Two aspects of your prior selection:
     - Subjective: **Informative Prior** ... your belief based Prior
       - __conjugate prior__
         - a class of distributions that present the same parametric form of the likelihood and their choice is frequently related to mathematical convenience and the likelihood. 
         
     - Objective: **Non-Informative (vague) Prior** when there is no information about the problem at hand.  
       - __Flat prior__
         - Uniform, Normal with huge variance, etc. The use of a flat prior typically yields results which are not too different from conventional statistical analysis.
       - __Improper prior__ 
         - It, in their parametric space, **does not integrate to 1**. For instance, in some cases Jeffery's priors are improper, but the posterior distribution is proper.
         - Jeffery's prior is proportional to the Fisher Information, which is the expected value of the second derivative of the log-likelihood function with respect to the parameter. Although it is non-informative, improper prior, the Fisher Information quantifies the variability of the parameter based on the available data. That is, the higher the value of the Fisher Information, the more concave is the log-likelihood, thus evidencing that the data helps to estimate the quantity of interest.
           - *He argues that any "non-informative prior" should be invariant to the parameterization(transformation) that we are using. If we create a `prior` that is proportional to the `Sqrt(FisherInf)` then the `prior` is **invariant** to the parameterization used.  
           <img src="https://user-images.githubusercontent.com/31917400/69545638-17f42080-0f8a-11ea-9d69-b686a3bfda57.jpg"/>

       - __Non-conjugate prior__ 
         - When the posterior distribution does not appear as a distribution that we can simulate or integrate. 
         - It makes the posterior to have an Open-form, but Metropolis-Hasting of MCMC solves the problem. 
   - why a paricular prior was chosen? 
     - The reality is that many of these prior distributions are making assumptions about the **`type of data`** we have.
     - There are some distributions used again and again, but the others are special cases of these dozen or can be created through a clever combination of two or three of these simpler distributions. A prior is employed because the assumptions of the prior match what we know about the **parameter generation process**. *Actually, there are multiple effective priors for a particular problem. A particular prior is chosen as some combination of `analytic tractability` + `computationally efficiency`, which makes other recognizable distributions when combined with popular likelihood functions. 
     - Examplary Prior Distributions
       - __Uniform Prior__
         - `Beta(1,1)` = Unif(0,1)
         - Whether you use this one in its continuous case or its discrete case, it is used for the same thing: 
           - `You have a set of events that are equally likely`. 
             - ex) see "binomial likelihood" case. Unif(0,1) says `θ` can be any value (ranging from 0 to 1) for any X.  
         - Note, the uniform distribution from ∞ to −∞ is not a probability distribution. 
           - Need to give **lower** and **upper** bounds for our values.
           - Not used as often as you’d think, since its rare we want hard boundaries on our values.
           
       - __Gaussian Prior__
         - Taking a **center** and **spread** as arguments, it states that 67% of your data is within 1*SD of the center, and 95% is within 2*SD. 
           - No need to check our value boundaries. 
         - coming up a lot because if you have multiple signals that come from any distribution (with enough signals), their average always converges to the normal distribution. `hist(np.array([np.mean(your_distribution) for i in range(your_samples)]))`.

       - __Beta Prior__ [0,1]
       
       - __Gamma Prior__ [0,∞]
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
       - __Heavy-tailed Prior__
         - The major advantage of using a heavy-tail distribution is it’s more robust towards outliers (we cannot be too optimistic about how close a value stays near the mean..)..let's start to care outliers..
         - `t-distribution` can be interpretted as the distribution over a **sub-sampled population** from the normal distribution sample. Since here our sample size is so small, atypical values can occur more often than they do in the general population. As our sub-population grows, the t-distribution becomes the normal distribution. 
           - The t-distribution can also be generalized to not be centered at 0.
           - The parameter `ν` lets you state how large you believe this subpopulation to be.
         - `Laplace-distribution` as an interesting modification to the normal distribution(replacing `exp(L2-norm)` with `exp(L1-norm)` in the formula). A Laplace centered on 0 can be used to put a strong **sparsity prior** on a variable while leaving a heavy-tail for it if the value has strong support for another value. 

## [b] `Likelihood`: MLE (Parameter Point Estimation)
 - `P( Data | θ )` is called likelihood of data given model parameters. The goal is to maximize the **likelihood function probability** `L(x,x,x,x..|θ)` to choose the best θ.
 <img src="https://user-images.githubusercontent.com/31917400/65486881-8c80e500-de9d-11e9-9d6b-e8d7b8af1d09.jpg"/>
  
   - **The formula for likelihood is model-specific**. 
   - People often use likelihood for evaluation of models: a model that gives higher likelihood to real data is better.
   - If one also takes the prior into account, then it’s maximum a posteriori estimation (MAP). `P(Data|θ)` x `P(θ)`. What it means is that, the likelihood is now weighted with some weight coming from the prior. MLE and MAP are the same if the **prior is uniform**.

## [c] `Posterior`: MAP (Parameter Point Estimation) 
 - `P( θ | Data )`, a posterior, is what we’re after. It’s a parametrized distribution over model parameters obtained from prior beliefs and data. The goal is to maximize the **posterior probability** `L(x,x,x,x..|θ)*P(θ)` that is the `value x Distribution` to choose the best θ.
 <img src="https://user-images.githubusercontent.com/31917400/66209863-49e6b600-e6b0-11e9-8668-aa2ccb4501e0.jpg"/>

   - we assume the model - Joint: `P(θ, Data)` which is `P(Data|θ)` x `P(θ)`
   - MAP can unlike MLE, avoid overfitting. MAP gives you the **`L2 Regularization`** term.  
 - But we still anyhow prefer to obtain Full Distribution rather than just point estimate. We want to address the uncertainty.
 - They are similar, as they compute a single estimate, instead of a full distribution.

### *c-1) Bayesian `Inference` (Parameter Full Distribution Estimation) 
 - "Inference" refers to how you learn parameters of your model. Unlike MLE and MAP, **Bayesian inference** means that it fully calculates the posterior probability distribution, hence the output is not a `single value` but a `pdf or pmf`.   
 - It's complex since we now have to deal with the **Evidence**(with the integral computation). But if we are allowed to use conjugation method, we can do **Bayesian inference** since it's easy. However, it’s not always the case in real-world applications. We then need to use MCMC or other algorithms as a substitute for the direct integral computation.
 - There are three main flavours: 
   - **0. Conjugation method**
     - Find a conjugate prior(very clear) based on the given likelihood then compute posterior using math! 
     - It simply implies the **integral of the `joint`** is a **closed form**! 
   - **1. MCMC:** a gold standard, but slow. We still need a prior? Yes! even fake prior because we still need the `joint`!
     - It implies the **integral of the `joint`** is an **open form**!
     - Obtain a posterior by sampling from the "Envelop".
   - **2. Variational inference:** faster but less accurate. Its drawback is that it’s model-specific..(use when likelihood & prior is `clear`)
     - It implies the **integral of the `joint`** is an **open form**!
     - Obtain a posterior by "appropriating other distribution".
 - If you have a truly infinite computational budget, MCMC should give more accurate solution than Variational Inference that trades some accuracy for speed. With a finite budget (say 1 year of computation), Variational Inference can be more accurate for very large models, but if the budget is large enough MCMC should give a better solution for any model of reasonable size.

### c-2) Variational Inference
Variational inference seeks to approximate the true posterior with an **approximate variational distribution**, which we can calculate more easily. The difference of EM-algorithm and Variational-Inference is the kind of results they provide; **`EM is just a point while VI is a distribution`.** However, they also have similarities. EM and VI can both be interpreted as minimizing some sort of **distance** between the true value and our estimate. 
 - For **EM:** which is the **`Maximum-Likelihood`**
   - ...assign fake param -> develop soft fake village -> calculate weighted param from the village -> conclude the ML value of each village -> Repeat from the start until the ML value gets to the maximum...Now, finally you obtain the best param. 
 - For **VI:** which is the **`Kullback-Leibler divergence`**.
> The term **variational** comes from the field of variational calculus. Variational calculus is just calculus over functionals instead of functions. Functionals are just a function of function(inputs a function and outputs a value). For example, the KL-divergence are functionals. The variational inference algorithms are simply optimizing functionals which is how they got the name "variational Bayes".

Set up
<img src="https://user-images.githubusercontent.com/31917400/67643780-febc6d80-f912-11e9-9c2c-155158e79e86.jpg"/> 

 - We have perfect likelihood and prior. But we don't have Evidence. So the un-normalized posterior(joint) is always the starting point. 
 - The main idea behind variational methods is to pick a fake? posterior `q(z)` as a **family of distributions** over the `latent variables` with **its own variational parameters**. Go with the exponential family in general? This is your FINGERS!!!! 
 - Then,find the **setting of the best parameters** that makes `q(z)` close to the posterior of interest.
<img src="https://user-images.githubusercontent.com/31917400/67643910-47c0f180-f914-11e9-9f01-81355bfdeea6.jpg"/> Use `q(z)` with the **fitted parameters** as a proxy for the posterior to predict about future data or to investigate the posterior distribution of the hidden variables (Typically, the true posterior is not in the variational family). 
 - Typically, in the true posterior distribution, the **latent variables** are not independent given the data, but if we **restrict our family of variational distributions** to a distribution that **`factorizes over each variable in Z`** (this is called a **mean field approximation**), our problem becomes a lot easier. 
 - We can easily pick each variational distribution(V_i) when measured by **Kullback Leibler (KL) divergence** because we compare this `Q(Z)` with our `un-normalized posterior` that we already have (KL divergence formula has a sum of terms involving V, which we can minimize...So the estimation procedure turns into an optimization problem). Once we arrive at the best `V*`, **we can use `Q(Z|V*)` as our best guess at the posterior**.  
<img src="https://user-images.githubusercontent.com/31917400/67644689-9f169000-f91b-11e9-88e7-55bde74becbf.jpg"/>

## KL-Divergence helps estimate the `z` that minimizes the distance b/w `Q(z)` and `P*(z)`
 - Step_01: Select the family distribution **Q** called a "variational family": a pool of **Q**
 - Step_02: Try to approximate the **full posterior** `P*(z)` with some variational distribution `Q(z)` by searching the best matching distribution, minimizing "KL-divergence" value.
   - minimizing KL-divergence value(E[log Q over P]) between `Q(z)` and `P*(z)`
 - Kullback Leibler-Divergence measures the difference(distance) b/w two distributions, so we minimize this value between your **variational distribution choice** and the **un-normalized posterior** (not differ from normalized real posterior...coz the evidence would become a constant...in the end.)   
<img src="https://user-images.githubusercontent.com/31917400/67668071-22110800-f967-11e9-8431-f424e819f18e.jpg"/>
 
If you additionally require that the **variational distribution factors completely over your parameters**, then this is called the variational mean-field approximation. 
 - Step_01: Select the family distribution **Q** called a "variational family" by **product of** `Q(z1)`, `Q(z2)`,...where z is the latent variable.  
 - Step_02: Try to approximate the **full posterior** `P*(z)` with some variational distribution `Q(z)` by searching the best matching distribution, minimizing "KL-divergence" value.
   - minimizing KL-divergence value(E[log Q over P*]) between `Q(z)` and `P*(z)`
<img src="https://user-images.githubusercontent.com/31917400/67224682-ab857f00-f429-11e9-9c21-af5503ea8c3a.jpg"/>

### c-3) Variational Inference + Neural Network = Scalable VI
10 years ago, people used to think that Bayesian methods are mostly suited for small datasets because it's computationally expensive. In the era of Big data, our Bayesian methods met deep learning, and people started to make some mixture models that has neural networks inside of a probabilistic model. 

How to scale Bayesian methods to `large datasets`? The situation has changed with the development of **stochastic Variational Inference**, trying to solve the inference problem exactly without the help of sampling. 

## > Background: Let's say we have a trouble with EM...

This is the useful story when you cannot calculate the MLE value in the EM algorithm..
<img src="https://user-images.githubusercontent.com/31917400/86511165-45d9f700-bdee-11ea-8e61-6e8adc5d0b8d.jpg"/>
When MLE does not work for the original margin of log-likelihood, then we try to get a **lower bound** with the function that we can easily optimize?  Instead of maximizing the original margin of log-likelihood, we can maximize its **lower bound**!!

## Now, Let's find the Lower Bound to estimate the `MLE value`!
> [note] But it's just a lower bound.. there is no guarantee that it gives us the correct parameter estimation! 
 - Perhaps we can try...a **family of lower bounds**?? i.e. try **many different lower bounds**!
 - ## Let me introduce `q(t)` as the variational distribution of the `alpha coefficient` (probability of the hidden membership `t`= c). These each fingers are the "lower boundsss"..Any distribution can be estimated by such "lower bounds"!!!
 - Make a bunch of different `lower bounds`:
   - Use (0)`Hidden "t" value`, and (1)`Alpha Coefficient: q(t)`, (2) **log(**`p(x, t)/q(t)`**)**
   - **min{** `q(t)`*log[`p(x,t)/q(t)`] **}** ...KL-Divergence.. This is the Jensen's lower bound? Let's re-express our `log marginal` with KL-Divergence elements!
 <img src="https://user-images.githubusercontent.com/31917400/71264042-20e2d280-233b-11ea-9b2e-e33f3d275411.jpg"/>

General EM-Algorithm
<img src="https://user-images.githubusercontent.com/31917400/71264565-458b7a00-233c-11ea-88d6-e3316d5fef5b.jpg"/>
We built a lower bound on the local likelihood which depends both on the theta to maximize the local likelihood and the parameter q which is the variational distribution value, and it suggests we can optimize this lower bound in iterations by repeating the two steps until convergence. On the E-step, fix theta and maximize the lower bound with respect to q. And on the M-step, fix q and maximize the lower bound with respect of theta. So this is the general view of the expectation maximization. 

### EX> Variational Autoencoder and Generative model: 
How can we perform efficient inference and learning in directed probabilistic models, in the presence of **continuous latent variables** with **intractable posterior distributions**, and **large datasets**? 

In contrast to the plain autoencoders, it has `sampling inside` and has `variational approximations`. 
 - for Dimensionality Reduction
 - for Information Retrieval
   
> [INTRO]: Why fitting a certain distribution into the disgusting DATA (**why do you want to model it**)?
 - If you have super complicated objects like natural images, you may want to build a probability distribution such as "GMM" based on the dataset of your natural images then try to generate **new complicated data**...
 - Application?
   - __Detect anomalies, sth suspicious__ 
     - ex> For example, you have a bank and you have a sequence of transactions, and then, if you fit your probabilistic model into this sequence of transactions, for a new transaction you can predict how probable this transaction is according to our model, our current training data-set, and if this particular transaction is not very probable, then we may say that it's kind of suspicious and we may ask humans to check it.
     - ex> For example, if you have security camera footage, you can train the model on your normal day security camera, and then, if something suspicious happens then you can detect that by seeing that some images from your cameras have a low probability of your image according to your model. 
   - __Deal with N/A__
     - ex> For example, you have some images with obscured parts, and you want to do predictions. In this case, if you have P(X) - probability distribution of your data -, it will help you greatly to deal with it. 
   - __Represent highly structured data in low dimensional embeddings__
     - ex> For example, people sometimes build these kind of latent codes for molecules and then try to discover new drugs by exploring this space of molecules in this latent space.....?? 

> Let's model the image!
<img src="https://user-images.githubusercontent.com/31917400/71101742-24495300-21af-11ea-9821-a14e07c54148.jpg"/>

 - [1.CNN]: Let's say that **CNN** will actually return your **logarithm of probability**. 
   - The problem with this approach is that you have to normalize your distribution. You have to make your distribution to sum up to one, with respect to sum according to all possible images in the world, and there are billions of them. So, this normalization constant is very expensive to compute, and you have to compute it to do the training or inference in the proper manner. HOW? You can use the chain rule. `Any probabilistic distribution can be decomposed into a product of some conditional distributions`, then we build these kind of conditional probability models to model our `overall joint probability`. 
 - [2.RNN]: how to represent these `conditional probabilities` is with **RNN** which basically will read your image pixel by pixel, and then outputs your prediction for the next pixel - Using proximity, Prediction for brightness for next pixel for example! And this approach makes modeling much easier because now normalization constant has to think only about 1D distribution.
   - The problem with this approach is that you have to generate your new images one pixel at a time. So, if you want to generate a new image you have to first generate X1 from the marginal distribution X1, then you will feed this into the RNN, and it will output your distribution on the next pixel and etc. So, no matter how many computers you have, one high resolution image can take like minutes which is really long...
 - [3.CNN with Infinite continuous GMM]: We can try an **infinite mixture of Gaussians** which can represent any probability distribution! Each object (image X) has a corresponding latent variable "T", and the image X is caused by this "T", so we can marginalize out w.r.t "T". And the conditional distribution `P(X|T)` is Gaussian. We can have a mixture of infinitely many Gaussians, for each value of "T"(membership), there's `one Gaussian` and we mix them with weights.   
   - ## Decoder?
   Only if we have `hidden variables`...
   <img src="https://user-images.githubusercontent.com/31917400/72342676-f3344380-36c4-11ea-90a2-ea05caf2e11a.jpg"/>
 
   - ## overview: get some `Q(latent variable)`
   <img src="https://user-images.githubusercontent.com/31917400/73172457-fd116a00-40fa-11ea-879e-33e83d73730e.jpeg"/>
        
   - Story: 
     - a. **Encoding**: Discover the memberships from our dataset -> b. **Decoding**: Generate new data based on the memberships

     - ## How to get `w` for Decoding? 
       - ## we need `t` for Encoding...**You should deal with the latent variable first!**
       - Before `w`, we need `P(t|x)`. Find the posterior of the `continuous latent variable "t"`
         - using MCMC to sample from P(t ∣ X, w)...?
         - using **Variational Inference**...? YES, let's try! First, think **How "t" is distributed**? 
         - **Step 1. Bring up the "factorized" variational distribution `q(t)`** and address a parameterization -`m`,`s`- via NN.
           - Assuming each `q(t)` as the Exponential family function with new Gaussian parameters - `m`vector, `s^2`matrix. 
           - Maximizing the likelihood function of our model w.r.t `m`,`s^2`...but are they clear? 
         - We can make `q(t)` more flexible. If assume all `q(t)` share the same parameterization - func`m()`, func`s^2()`, depending on individual parameter `x` and `weight`.. then the training get easier. We have the original input data `x` so let's get some weight `φ` via CNN!
           <img src="https://user-images.githubusercontent.com/31917400/72226055-8a45b200-3584-11ea-96ce-b6ad7d78de6f.jpg"/>
           
         - **Step 2. Build an AutoEncoder**
           - To get the Jensen's lower bound at the end, we pass our **initial dataset** through the `first neural network` encoder with parameters`φ` to get the parameters `m`,`s^2` of the variational distribution `q(t)` to get the **latent variable** disribution. 
           - We MC sample from this distribution`q(t)` random data pt `t`.  
           - We pass this sampled vector `T` into the `second neural network` with parameters`w` 
             - It outputs us the distribution that are as close to the input data as possible.
           <img src="https://user-images.githubusercontent.com/31917400/72226599-d136a600-358a-11ea-9e13-69138c206a53.jpg"/>
           
 - ### Next, two CNN for Φ and w: Maximize Jensen's Lower bound
   - __[Note in Encoder]: Outlier Detection__???? for a new image which the network never saw, of some suspicious behavior or something else, our conditional neural network of the encoder can output your **latent variable distribution** as far away from the Gaussian. By looking at the distance between the variational distribution `q(t)` and the standard Gaussian, you can understand how anomalistic a certain point is ... they are outliers.    
   <img src="https://user-images.githubusercontent.com/31917400/72226852-bca7dd00-358d-11ea-98d6-20965d0dce46.jpg"/>
   
   - __Gradient of Encoder:__ Make an Expected Value ?
     - we're passing our image through our Encoder, and compute the **usual gradient** of this first neural network with respect to its parameters `Φ` to get the parameters(Φ) of the variation distribution `q(t|Φ)`. We use **"log derivitive trick"** to approximate the gradient (make the form of expected value?) but it has some problem: `the variance of this stochastic approximation will be so high that you will have to use lots and lots of gradients to approximate this thing accurately`. How can we estimate this gradient with a much **smaller variance estimate**?  
   - __Gradient of Decoder:__ Make an Expected Value ?
     - we sample `t` from the variation distribution `q(t|Φ)` and put this `point` as input to the Decoder with parameters `w`. And then we just compute the **usual gradient** of this second neural network with respect to its parameters `w`.  
   <img src="https://user-images.githubusercontent.com/31917400/72433990-7a9bb880-3792-11ea-8cfd-f3e6778fa8ad.jpg"/>
   
   - __Issues of gradient of Encoder:__ 허벌창 그라디언트여? How can we better estimate this varying gradient with a much **smaller variance estimate**?  
     - 왜 허벌창? our input data (x) is 이미지니깐...
     - when sampling `t`, **"reparameterization trick"** of our latent variable makes the a Jensen's lower bound estimator easy to be optimized using standard stochastic gradient.
     - so..you just sample from a identity matrix...All works will be done by `m` and `s^2`..
   <img src="https://user-images.githubusercontent.com/31917400/73176973-afe6c580-4105-11ea-8822-49b2d202c156.jpg"/>


### Learning with priors
<img src="https://user-images.githubusercontent.com/31917400/69436481-5b0b8500-0d39-11ea-8e3d-1d565674042e.jpg"/>

### EX> Variational Dropout and Scalable BNN
Compress NN, then fight severe overfitting on some complicated datasets. 
 
We first pick a fake? posterior `q(z|v)` as a **family of distributions** over the `latent variables` with **its own variational parameters**`v`. KL-divergence method helps us to minimize the distance between `P(z)` and `q(z)`, and in its optimization process, we can use `mini-batching` training strategy(since its likelihood can be split into many pieces of log sum), which means we don't need to compute the whole training of the likelihood. ELBO supports mini-batching.    
 - We can use MonteCarlo estimates for computing stochastic gradient, which is especially useful when the reparameterization trick for `q(z|v)` is applicable. 
 
????????????????????????????????????????????????????????????????? 



## [d] Data value `Prediction` 
Evidence is discussed in the process of inference (not in the prediction...?) Bayesian methods are appealing for prediction problems thanks to their ability to naturally incorporate both **`sample variability`** and **`parameter uncertainty`** into a predictive distribution. Let's train data points X and Y. We want predict the new Y at the end. In Bayesian Prediction, the predicted value is a **weighted average** of output of our model for all possible values of parameters. 
<img src="https://user-images.githubusercontent.com/31917400/75678832-62babe00-5c86-11ea-8efa-0831cbc00227.jpg"/>

### Real-time data? 
Alternative perspective on the prediction method is **Bayesian Prediction with Copulas**. Handling **data arriving in real time** requires a flexible non-parametric model, and the Monte Carlo methods necessary to evaluate the predictive distribution in such cases can be too expensive to rerun each time new data arrives. With respect to this, Bayesian Prediction with Copulas' approach facilitates the prediction **`without computing a posterior`**. 

 - ### Concept 01> Recursive nature of the updates in predictive distribution
   <img src="https://user-images.githubusercontent.com/31917400/77644983-de441e00-6f59-11ea-8ce8-f25d4931b4c9.jpg"/> However, in cases where **it is not possible to work directly with the posterior**, this natural Bayesian updating formula is out of reach.

 - ### Concept 02> Let's work directly with the posterior: DPMixture of Gaussian? Beta? some "kernel"? 
   In our context of estimating the predictive distribution in real time, it is not possible to look at the entire dataset all at once, thus we seek the flexibility of a non-parametric model, largely to avoid potential model misspecification. That is, it is necessary to start with a sufficiently flexible model that can adapt to the shape of the distribution as they arrive. In these non-parametric cases, θ is not a finite-dimensional parameter, but it is an infinite-dimensional index - ![formula](https://render.githubusercontent.com/render/math?math=\mu_k?,\sigma_k?,\nu_k?) - of the distribution clusters(Gaussian, Beta, whatever...)that explaining the dataset. The most common strategy, in the present context of modelling densities, is the so-called Dirichlet process mixture model. <img src="https://user-images.githubusercontent.com/31917400/77647494-6d533500-6f5e-11ea-9768-d5149655cab3.jpg"/> The problem is that given the posterior ![formula](https://render.githubusercontent.com/render/math?math=\pi_\n-1) based on the full data, when new data ![formula](https://render.githubusercontent.com/render/math?math=\x_n) arrives, the MCMC must be rerun on the full data to get the posterior ![formula](https://render.githubusercontent.com/render/math?math=\pi_n) or the predictive density ![formula](https://render.githubusercontent.com/render/math?math=\f_n). This can be prohibitively slow, thereby motivating a fast recursive approximation.

 - ### Concept 03> Gaussian Copula Density
   To circumvent the aforementioned computational difficulties in Bayesian updating in the predictive models, we turn to a new strategy: A Recursive Approximation with Copulas. A Copula as a mathematical object captures the joint behavior of **two different Random Variables**, each of which follows different distribution, and returns a single bivariate distribution formula. Sklar theorem implies that there exists a symmetric copula density ![formula](https://render.githubusercontent.com/render/math?math=\C_n) such that <img src="https://user-images.githubusercontent.com/31917400/77655759-bc9f6280-6f6a-11ea-88f9-a10e7b7e7c38.jpg"/> That is, for each Bayesian model, there exists a unique sequence {![formula](https://render.githubusercontent.com/render/math?math=\C_n)} of copula densities. This representation reveals that it is possible to directly and recursively update the predictive distribution without help of MCMC. It has the advantage of directly estimating the predictive density and does not require numerical integration to compute normalising constants.
   
   For a Dirichlet process mixture model, with `Gaussian kernel` - N(x|u, 1) - and `DP sample prior` - ![formula](https://render.githubusercontent.com/render/math?math=\alpha=\alpha,\G_0=\N(0,1/\tau)) where ![formula](https://render.githubusercontent.com/render/math?math=\rho=1/\tau). 
   - `u` = ![formula](https://render.githubusercontent.com/render/math?math=\F_n_1(x))
   - `v` = ![formula](https://render.githubusercontent.com/render/math?math=\F_n_1(X_n))
   
   The `Gaussian Copula Density` is <img src="https://user-images.githubusercontent.com/31917400/77661903-23c11500-6f73-11ea-8196-7b73cb9b3a9a.jpg"/> In particular, we consider the following recursive sequence of predictive densities <img src="https://user-images.githubusercontent.com/31917400/77659684-44d43680-6f70-11ea-93f8-27929110970f.jpg"/> But it's too complicate...
   On the CDF distribution function scale, the algorithm is a bit more transparent, that is, <img src="https://user-images.githubusercontent.com/31917400/77660719-96c98c00-6f71-11ea-8c28-c7c22ecf2ebe.jpg"/> 
   The take-away message is that there exists a recursive update of the predictive density ![formula](https://render.githubusercontent.com/render/math?math=\f_n) in the Dirichlet process mixture model formulation, characterised by a copula density. (???really???)
   
 - ### Add them up> "Recursive Algorithm"
   <img src="https://user-images.githubusercontent.com/31917400/77670548-460c6000-6f7e-11ea-832c-2822d7b9069a.jpg"/> The choice of `ρ` is entirely up to the discretion of the researcher, with values closer to 1 corresponding to less smoothing (ρ=0.90 is a reasonable choice?). For the weights, a choice like ![formula](https://render.githubusercontent.com/render/math?math=\alpha_i)=(i+1)^-r for r ∈ (0.5, 1]...as `i` grows, `α` decreases (r=1 as a default choice?). In choosing the initial guess of ![formula](https://render.githubusercontent.com/render/math?math=\F_0), try to capture the support of given dataset distribution? Since this predictive function is not sure (if there is little or no data to use as a guide), we go with some kernel density? such as t-distribution?? But we totally ignore DP prior or kernel likelihood????????    
```


```
---------------------------------------------------------------------------------------------------------
# 2. Modeling
In parametric method, we define a model that depends on some parameter "theta" and then we find optimal values for "theta" by taking MLE, or MAP. And as data becomes more and more complex, we need to add more and more parameters(think about LM's coefficients, linear? polynomial?) so we can say **the number of parameters are fixed**.
 - Fixed number of parameters => so the complexity is limited. 
 - **Fast Inference** coz you just simply feed the weights then the prediction would be just the scalar multiplication.
 - But training is complicated and takes time. 

In Non-parametric method, **the number of parameters depend on the dataset size**. That is, as the number of data points increases, the decision boundary becomes more and more complex. 
 - Not Fixed number of parameters => so the complexity is arbitrary.
 - **Slow Inference** coz you have to process all the data points to make a prediction. 
 - But training is simple coz it in most cases just remembers all points . 

__[Parametric]__
 - **A. Bayesian Network as PGM**
   - Bayesian Network is "Directed" and "Acyclic". It cannot have **interdependent** variables. 
   <img src="https://user-images.githubusercontent.com/31917400/66124100-7381dd80-e5db-11e9-9d5d-c37b07d2f447.jpg"/>

In the settings where data is scarce and precious and hard to obtain, it is difficult to conduct a large-scale controlled experiment, thus we cannot spare any effort to make the best use of available input. `With small data, it is important to **quantify uncertainty**` and that’s precisely what Bayesian approach is good at. In Bayesian Modeling, there are two main flavours:
 - **B. Statistical Modeling:** 
   - Multilevel/Hierarchical Modeling(Regression?)
 - **C. probabilistic Machine Learning approach:** using data for a computer to learn automatically from it. It outputs probabilistic predictions...that's why probabilistic.. also these probabilities are only statements of belief from a classifier.
   - __1) Generative modeling:__ One can sample or generate examples from it. Compare with classifiers(discriminative model to model `P(y|x)` to discriminate between classes based on x), **a generative model is concerned with joint distribution `P(y,x)`**. It’s more difficult to estimate that distribution, but **it allows sampling** and of course one can get `P(y|x)` from `P(y,x)`.
     - **LDA:** You start with a matrix where `rows` are **documents**, `columns` are **words** and `each element` is a **count of a given word** in a given document. LDA “factorizes” this matrix of size n x d into two matrices, documents/topics (n x k) and topics/words (k x d). you can’t multiply those two matrices to get the original, but since the appropriate rows/columns sum to one, **you can “generate” a document**. 
     
__[Non-Parametric]__
 - **A. Bayesian non-parametrics Modeling:** the number of parameters in a model can grow as more data become available. This is similar to SVM, for example, where the algorithm chooses support vectors from the training points. Nonparametrics include **Hierarchical Dirichlet Process** version of LDA(where the number of topics chooses itself automatically), and **Gaussian Processes**.
   - __1) Gaussian Processes:__ It is somewhat similar to SVM - both use **kernels** and have similar **scalability**(which has been vastly improved throughout the years by using approximations). 
     - A natural formulation for GP is **`regression`**, with **classification** as an afterthought. For SVM, it’s the other way around.
     - As most "normal" methods provide **point estimates**, "Bayesian" counterparts like GP also output **uncertainty estimates** while SVM are not. 
     - Even a sophisticated method like GP normally operates on an **assumption of homoscedasticity**, that is, "uniform noise" levels. In reality, noise might differ across input space (be heteroscedastic). 
     - GP outputs a mean curve and CI(cov) curves.
       
   - __2) Dirichlet Process:__ The infinite-dimensional generalization of the Dirichlet distribution is the Dirichlet process. In short, the Dirichlet Process is a generalization of Dirichlet distributions where a sample from DP generates a Dirichlet distribution. Interestingly, the generalization allows the Dirichlet Process to have an infinite number of components (or clusters), which means that there is no limit on the number of Hyper-parameters. Using DP, we sample proportion of each element in a vector or multinomial random variable from the undefined dimension that can go to infinity. 


### > Example 01. Linear Model..some different ways to address Coefficients and error!
 - a) Frequentist LM  
   - typically go through the process of checking the 1.`residuals against a set of assumptions`, 2.`adjusting/selecting features`, 3.`rerunning the model`, 4.`checking the assumptions again`.... 
     - Frequentist diagnose is based on the `fitted model` using **MLE** of the model parameters.
       - "likelihood": `f(x|β)`
       - "likelihood function": `L(x,x,x,x|β)` by fitting a distribution to the certain **data** so...producting them, then **differentiating** to get the best `β`. But the result is just a **point estimate**(also subject to the overfitting issue)...it cannot address **`Uncertainty`**!
       - subject to overfitting!
     
 - b) Bayesian Hierarchical LM 
   - It allows a useful mechanism to deal with insufficient data, or poorly distributed data. If we have fewer data points, the posterior distribution will be more spread out. As the amount of data points increases, the likelihood washes out the prior.  
   - It puts a prior on the coeffients and on the noise so that in the absence of data, the **priors can take over !**
   - Once fitting it to our data, we can ask:
     - What is the estimated `linear relationship`
       - what is the **confidence on that relation**, and the **full posterior distribution on that relation**?
     - What is the estimated `noise` and the **full posterior distribution on that noise**?
     - What is the estimated `gradient` and the **full posterior distribution on that gradient**?

 - __Posterior Computation by Bayesian Inference:__ How to avoid computing the Evidence?
   - A> When we want to get the model parameter, the Evidence is always a trouble. There is a way to avoid `computing the **Evidence**`. In **MAP**, we don't need the "Evidence". But the problem is that we cannot use its result as a prior for the next step since the output is a single point estimate. 
     - Below is MAP for LM parameter vector `w`.
     - The result says it's the traditional **MLE** value + `L2 regularization` term (because of the prior) that fix overfitting.
     - But it still does not have any representation of **Uncertainty**!
   <img src="https://user-images.githubusercontent.com/31917400/66239444-ca79d680-e6f1-11e9-8e3d-c8d009647fac.jpg"/>

   - B> There is another way to avoid `computing the **Evidence**` - Use **Conjugate prior**. We can, but do not need to compute the Evidence.  
     - Conjugate `Prior` as a member of certain family distributions, is conjugate to a `likelihood` if the resulting posterior is also the member of the same family. 
       - Discrete Likelihood
         - `Beta prior` is conjugate to Bernoulli likelihood. (so Bernoulli model? then choose Beta)
         - `Beta prior` is conjugate to Binomial likelihood. (so Binomial model? then choose Beta)
         - `Dirichlet prior` is conjugate to Muiltinomial likelihood. (so Multinomial model? then choose Dirichlet)
         - `Gamma prior` is conjugate to Poisson likelihood. (so Possion model? then choose Gamma)
         - `Beta prior` is conjugate to Geometric likelihood. (so Geometric model? then choose Beta)
       - Continous Likelihood
         - `Gaussian prior` is conjugate to **Gaussian likelihood + known SD**. 
         - `Inverse Gamma prior` is conjugate to **Gaussian likelihood + Known μ**. 
         - `Pareto prior` is conjugate to **Uniform likelihood**.
         - `Gamma prior` is conjugate to **Pareto likelihood**.
         - `Gamma prior` is conjugate to **Exponential likelihood**.
       - If the likelihood is a member of **Exponential-family**, it always guarantees the presence of the conjugate prior. 

     - Gaussian Prior for **Gaussian likelihood + known SD**     
     <img src="https://user-images.githubusercontent.com/31917400/66254604-8da0f480-e770-11e9-88d4-b5686ba7c91d.jpg"/>
  
     - Now we can **take advantage of having access to the full posterior distribution of the model parameter(Coefficient)**: we can either obtain a point estimator from this distribution (e.g. posterior mean, posterior median, ...) or conduct the same analysis using this estimate...now we can say **`Uncertainty`**.  
     - Check the goodness of fit of the estimated model based on the predictive residuals. It is possible to conduct the same type of diagnose analysis of Frequentist's LM. 

   - C> To approximate the posterior, we use the technique of drawing random samples from a posterior distribution as one application of Monte Carlo methods. 
     - 1. Specify a prior `π(β)`.
     - 2. Create a model mapping the training inputs to the training outputs.
     - 3. Have a MCMC algorithm draw samples from the posterior distributions for the parameters. 
     









---------------------------------------------------------------------------------------------------------
# 3> Model Comparison
























































