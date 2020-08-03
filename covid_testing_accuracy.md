# COVID 19 Example

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

[Questions about COVID19 Accuracy](https://www.nbcnews.com/health/health-news/questions-about-covid-19-test-accuracy-raised-across-testing-spectrum-n1214981)

$$Pr(COVID19|positive) = \frac{Pr(positive|COVID19) Pr(COVID19)} {Pr(positive)}$$

$$Pr(positive) = Pr(positive|COVID19) Pr(COVID19) + Pr(positive|HEALTHY) (1 − Pr(COVID19))$$

```
PrPC = 0.9999  # probability of a positive test result given you have COVID19
PrPH = 0.0001  # probability of a positive test result if you do not have COVID19 
PrC = 0.001  # probability of having COVID19
PrP = PrPC * PrC + PrPH * (1-PrC)  # marginal probability of having a positive test
PrCP = PrPC * PrV / PrP  # use Bayes to estimate probability of having COVID19 given a positive test
PrCP
```
