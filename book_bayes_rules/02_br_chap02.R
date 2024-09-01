# Load packages
library(bayesrules)
library(tidyverse)
library(janitor)

# Import article data
data(fake_news)

fake_news %>% 
  tabyl(type) %>% 
  adorn_totals("row")

fake_news %>% 
  tabyl(title_has_excl, type) %>% 
  adorn_totals("row")

# Define possible articles
article <- data.frame(type = c("real", "fake"))

# Define the prior model
prior <- c(0.6, 0.4)

# Simulate 3 articles
sample_n(article, size = 3, weight = prior, replace = TRUE)

# Set the seed. Simulate 3 articles.
set.seed(84735)
sample_n(article, size = 3, weight = prior, replace = TRUE)

# Simulate 10000 articles.
set.seed(84735)
article_sim <- sample_n(article, size = 10000,
                        weight = prior, replace = TRUE)

ggplot(article_sim, aes(x = type)) +
  geom_bar()

article_sim %>% 
  tabyl(type) %>% 
  adorn_totals()

article_sim <- article_sim %>% 
  mutate(data_model = case_when(type == "fake" ~ 0.2667,
                                type == "real" ~ 0.0222))

glimpse(article_sim)

# Define whether there are exclamation points
data <- c("no", "yes")

# Simulate exclamation point usage
set.seed(3)
article_sim <- article_sim %>%
  group_by(1:n()) %>% 
  mutate(usage = sample(data, size=1,
                        prob = c(1 - data_model, data_model)))

article_sim %>% 
  tabyl(usage, type) %>% 
  adorn_totals(c("col", "row"))

ggplot(article_sim, aes(x = type, fill = usage)) +
  geom_bar(position = "fill")

ggplot(article_sim, aes(x = usage)) +
  geom_bar()

article_sim %>% 
  filter(usage == "yes") %>% 
  tabyl(type) %>% 
  adorn_totals("row")

ggplot(article_sim, aes(x = type)) +
  geom_bar() +
  facet_wrap(~ usage)

# Load the data
data(pop_vs_soda)

# Summarize pop use by region
pop_vs_soda %>% 
  tabyl(pop, region) %>% 
  adorn_percentages("col")

pop_vs_soda %>% 
  tabyl(pop) %>% 
  adorn_percentages("col")

# Define possible win probabilities
chess <- data.frame(pi = c(0.2, 0.5, 0.8))

# Define the prior model
prior <- c(0.10, 0.25, 0.65)

# Simulate 10000 values of pi from the prior
set.seed(84735)
chess_sim <- sample_n(chess, size = 10000, weight = prior, replace = TRUE)

# Simulate 10000 match outcomes
chess_sim <- chess_sim %>% 
  mutate(y = rbinom(10000, size = 6, prob = pi))

# Check it out
chess_sim %>% 
  head(3)

#Summarize the prior
chess_sim %>% 
  tabyl(pi) %>% 
  adorn_totals("row")

# Plot y by pi
ggplot(chess_sim, aes(x = y)) +
  stat_count(aes(y = after_stat(prop))) +
  facet_wrap(~ pi)

