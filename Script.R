################################################################################
#title: "MovieLens Recommendation System Project"
#author: "Ringgold P. Atienza"
################################################################################

#Install/load packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(forcats)) install.packages("forcats", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

#MovieLens 10M dataset: http://files.grouplens.org/datasets/movielens/ml-10m.zip

#create dl tempfile
dl <- tempfile()

#download dataset (zipfile)
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#unzip ratings data
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

#unzip movies data
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

#set column names
colnames(movies) <- c("movieId", "title", "genres")

#set maxtrix as dataframe
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

#left_join movies data on ratings data
movielens <- left_join(ratings, movies, by = "movieId")

movieId <- unique(movielens$movieId)

################################################################################
#Mutate dataset to flesh out relevant variables for the model

#Data manipulation: Extract the year of the release of the movie
movielens <- mutate(movielens, title = str_trim(title)) %>%
  extract(title, c("title_temp", "movieYear"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", 
          remove = F) %>%
  mutate(movieYear = if_else(str_length(movieYear) > 4, 
                             as.integer(str_split(movieYear, 
                                                  "-", simplify = T)[1]), 
                             as.integer(movieYear))) %>%
  mutate(title = if_else(is.na(title_temp), title, title_temp)) %>%
  select(-title_temp)

#Check for missing values
na_count <- sapply(movielens, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count

#Data manipulation: Mutate timestamp into dates and years
movielens <- mutate(movielens, reviewDate = round_date(as_datetime(timestamp), unit = "week"))
movielens <- mutate(movielens, reviewYear = year(as_datetime(reviewDate)))
movielens <- subset(movielens, select = -c(timestamp))

#Data manipulation: Create age of the movie during review variable
movielens <- mutate(movielens, movieAge = reviewYear - movieYear)

#Data manipulation: add no. of times users rate movies
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
movielens <- ddply(movielens, .(userId), transform, user = count(userId))
movielens <- subset(movielens, select = -c(user.x))
setnames(movielens, "user.freq", "userFreq")

#Data manipulation: add mo.of times movies are rated
movielens <- ddply(movielens, .(movieId), transform, movie = count(movieId))
movielens <- subset(movielens, select = -c(movie.x))
setnames(movielens, "movie.freq", "movieFreq")

detach(package:plyr) #unload plyr package as it can cause compatibility issues in other packages.

################################################################################
#Partition movielens data into train and test set
set.seed(2022, sample.kind = "Rounding") #Set.seed is important for the reproducibility of this entire project
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.2, list = FALSE)
train <- movielens[-test_index,]
temp <- movielens[test_index,]

#Make sure userId and movieId in test set are also in train set
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

#Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

rm(dl, ratings, movies, test_index, temp, removed, na_count)

################################################################################
#Create validation set for model tuning
validation_index <- createDataPartition(y = train$rating, times = 1, p = 0.2, list = FALSE)
trainset <- train[-validation_index,]
temp <- train[validation_index,]

#To overlap the movie and user Ids using semi_join
validation <- temp %>%
  semi_join(trainset, by = "movieId") %>%
  semi_join(trainset, by = "userId")

rm(validation_index, temp)

################################################################################
#Summaries and visual inspection of the variables

#Summarize data set (first 10 rows)
head(train, 10)
summary(train)

#Summarize number of users, movies, and ratings
train %>% summarize(n_users = n_distinct(userId),
                  n_movies = n_distinct(movieId),
                  n_ratings = nrow(train))

#Create plot theme
plot_theme <- theme(plot.title = element_text(size = rel(1.5)),
                    plot.caption = element_text(size = 12, face = "italic"), 
                    axis.title = element_text(size = 12))

#Plot Figure 1. Actual rating distribution
ggplot(train, aes(rating)) +
  geom_bar(stat = "count", color = "black", ) +
  labs(x = "Rating", y = "Count",
       subtitle = "n = 8,000,074 ratings",
       caption = "*based on training set") + 
  plot_theme

#Plot Figure 2. Distribution of average ratings per movie
train %>% group_by(movieId) %>%
  summarise(ave_rating = sum(rating)/n()) %>%
  ggplot(aes(ave_rating)) +
  geom_histogram(binwidth = .10, color = "black") +
  labs(x = "Average rating per movie", y = "Number of movies",
       subtitle = "n = 10,677 movies",
       caption = "*based on training set") + 
  plot_theme

#Plot Figure 3. Distribution of average ratings by user
train %>% group_by(userId) %>%
  summarise(ave_rating = sum(rating)/n()) %>%
  ggplot(aes(ave_rating)) +
  geom_histogram(binwidth = .10, color = "black") +
  labs(x = "Average rating per user", y = "Number of users",
       subtitle = "n = 69,878 users",
       caption = "*based on training set") + 
  plot_theme

#Plot Figure 4. Distribution of ratings by movie year
ggplot(train, aes(movieYear)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_y_continuous(breaks = seq(0, 8000000, 100000), labels = seq(0, 8000, 100)) +
  labs(x = "Movie's Year of Release", y = "Count ('000s)", caption = "*based on train dataset") +
  plot_theme

#Plot Figure 5. Distribution of rating by movie age
ggplot(train, aes(movieAge)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_y_continuous(breaks = seq(0, 1100000, 100000), labels = seq(0, 1100, 100)) +
  labs(x = "Movie's Age", y = "Count (,000s)", caption = "*based on train datase") +
  plot_theme

#Plot Figure 6. Distribution of rating by genre
train %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  ggplot(aes(x = fct_infreq(genres))) +
  geom_bar() +
  scale_y_continuous(breaks = seq(0, 4000000, 500000)) +
  labs(x = "Genre", y = "Count", caption = "*based on train dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12),
        axis.text.x = element_text(angle = 60, hjust =1))

#Plot Figure 6.5 Ratings per genre
train %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarise(m = mean(rating)) %>%
  ggplot(aes(y = m, x = reorder(genres, -m), label=sprintf("%0.2f", round(m, digits = 2)))) +
  geom_point(size = 11) +
  geom_segment(aes(x = genres, xend = genres, y = 0, yend = m)) +
  geom_text(color = "white", size = 3) +
  ylim(0, 5) +
  labs(x = "Genre", y = "Rating", caption = "*based on train dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12),
        axis.text.x = element_text(angle = 60, hjust =1))

################################################################################
#Globalization of Global Effects
################################################################################
#Baseline predictors

#Setting Loss Function (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
  }

#Baseline predictors bui = µ +bu +bi
#1st find the lamba value as the regularization term. This avoids overfitting by penalizing least square estimate.

#Choose penalty terms using cross-validation
lambdas <- seq(0, 10, 0.25)

mu <- mean(trainset$rating)

#Penalty term for movie effect
moviesum <- trainset %>% 
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>% 
    left_join(moviesum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})

#Plot 7. Penalty Terms for Movie Effect
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

lambda_bi <- lambdas[which.min(rmses)]

#Penalty term for user effect
usersum <- trainset %>% 
  group_by(userId) %>%
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>% 
    left_join(usersum, by='userId') %>% 
    mutate(b_u = s/(n_i+l)) %>%
    mutate(pred = mu + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})

#Plot 8. Penalty Terms for User Effect
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

lambda_bu <- lambdas[which.min(rmses)]

rm(lambdas, rmses, moviesum, usersum)

#Predict (mu + b_i + b_u)
movie_avgs <- trainset %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + lambda_bi))
user_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n() + lambda_bu))
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, validation$rating)
rmse_user_step <- data.frame(Stepwise = "Global Average + Movie effect + User Effect", 
                             RMSE = rmse_user_step_temp,
                             Difference = 0)

#Baseline Prediction + User Frequency
userfreq_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu - b_i - b_u))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  mutate(pred = mu + b_i + b_u + b_uf) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_user_step_temp <- data.frame(Stepwise = "+ User Frequency", 
                             RMSE = rmse_user_step_temp,
                             Difference = rmse_user_step[1,2] - rmse_user_step_temp)

rmse_user_step <- rbind(rmse_user_step, rmse_user_step_temp)

#Baseline Prediction + Movie Age
movieage_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(movieAge) %>% 
  summarize(b_ma = mean(rating - mu - b_i - b_u))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  mutate(pred = mu + b_i + b_u + b_ma) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_user_step_temp <- data.frame(Stepwise = "+ Movie Age", 
                                  RMSE = rmse_user_step_temp,
                                  Difference = rmse_user_step[1,2] - rmse_user_step_temp)

rmse_user_step <- rbind(rmse_user_step, rmse_user_step_temp)

#Baseline Prediction + Movie Year
movieyear_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(movieYear) %>% 
  summarize(b_my = mean(rating - mu - b_i - b_u))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  mutate(pred = mu + b_i + b_u + b_my) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, validation$rating) #0.8658615
rmse_user_step_temp <- data.frame(Stepwise = "+ Movie Year", 
                                  RMSE = rmse_user_step_temp,
                                  Difference = rmse_user_step[1,2] - rmse_user_step_temp)

rmse_user_step <- rbind(rmse_user_step, rmse_user_step_temp)

#Baseline Prediction + Genre Effect
genres_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_user_step_temp <- data.frame(Stepwise = "+ Genres Effect", 
                                  RMSE = rmse_user_step_temp,
                                  Difference = rmse_user_step[1,2] - rmse_user_step_temp)

rmse_user_step <- rbind(rmse_user_step, rmse_user_step_temp)

#Baseline Prediction + Movie Frequency
moviefreq_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(movieFreq) %>% 
  summarize(b_mf = mean(rating - mu - b_i - b_u))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_user_step_temp <- data.frame(Stepwise = "+ Movie Frequency", 
                                  RMSE = rmse_user_step_temp,
                                  Difference = rmse_user_step[1,2] - rmse_user_step_temp)

rmse_user_step <- rbind(rmse_user_step, rmse_user_step_temp)
rmse_user_step

#Plot Figure 9. Baseline Prediction + 1 variable
rmse_user_step <- rmse_user_step [order(-rmse_user_step $RMSE),]

ggplot(rmse_user_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  plot_theme

################################################################################
#Stepwise modelling (baseline + movie frequency)
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf) %>%
  pull(pred)

rmse_mf_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_mf_step <- data.frame(Stepwise = "Baseline + Movie Frequency", 
                             RMSE = rmse_mf_step_temp,
                             Difference = 0)

#Add: Movie Age
movieage_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  group_by(movieAge) %>% 
  summarize(b_ma = mean(rating - mu - b_i - b_u - b_mf))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma) %>%
  pull(pred)

rmse_mf_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_mf_step_temp <- data.frame(Stepwise = "+ Movie Age", 
                                  RMSE = rmse_mf_step_temp,
                                  Difference = rmse_mf_step[1,2] - rmse_mf_step_temp)

rmse_mf_step <- rbind(rmse_mf_step, rmse_mf_step_temp)

#Add: Genre Effect
genres_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_mf))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_g) %>%
  pull(pred)

rmse_mf_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_mf_step_temp <- data.frame(Stepwise = "+ Genres Effect", 
                                RMSE = rmse_mf_step_temp,
                                Difference = rmse_mf_step[1,2] - rmse_mf_step_temp)

rmse_mf_step <- rbind(rmse_mf_step, rmse_mf_step_temp)

#Add: Movie Year
movieyear_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  group_by(movieYear) %>% 
  summarize(b_my = mean(rating - mu - b_i - b_u - b_mf))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_my) %>%
  pull(pred)

rmse_mf_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_mf_step_temp <- data.frame(Stepwise = "+ Movie Year", 
                                RMSE = rmse_mf_step_temp,
                                Difference = rmse_mf_step[1,2] - rmse_mf_step_temp)

rmse_mf_step <- rbind(rmse_mf_step, rmse_mf_step_temp)

#Add: User frequency
userfreq_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu - b_i - b_u - b_mf))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_uf) %>%
  pull(pred)

rmse_mf_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_mf_step_temp <- data.frame(Stepwise = "+ User Frequency", 
                                RMSE = rmse_mf_step_temp,
                                Difference = rmse_mf_step[1,2] - rmse_mf_step_temp)

rmse_mf_step <- rbind(rmse_mf_step, rmse_mf_step_temp)
rmse_mf_step

#Plot Figure 10. Baseline Prediction + 2 variables
rmse_mf_step <- rmse_mf_step [order(-rmse_mf_step $RMSE),]

ggplot(rmse_mf_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  plot_theme

################################################################################
#Stepwise modelling (baseline + movie frequency + movie age)
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma) %>%
  pull(pred)

rmse_ma_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_ma_step <- data.frame(Stepwise = "Baseline + Movie Frequency + Movie Age", 
                           RMSE = rmse_ma_step_temp,
                           Difference = 0)

#Add: User Frequency
userfreq_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu - b_i - b_u - b_mf - b_ma))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_uf) %>%
  pull(pred)

rmse_ma_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_ma_step_temp <- data.frame(Stepwise = "+ User Frequency", 
                                RMSE = rmse_ma_step_temp,
                                Difference = rmse_ma_step[1,2] - rmse_ma_step_temp)

rmse_ma_step <- rbind(rmse_ma_step, rmse_ma_step_temp)

#Add: Genre Effect
genres_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_mf - b_ma))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_g) %>%
  pull(pred)

rmse_ma_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_ma_step_temp <- data.frame(Stepwise = "+ Genres Effect", 
                                RMSE = rmse_ma_step_temp,
                                Difference = rmse_ma_step[1,2] - rmse_ma_step_temp)

rmse_ma_step <- rbind(rmse_ma_step, rmse_ma_step_temp)

#Add: Movie Year
movieyear_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  group_by(movieYear) %>% 
  summarize(b_my = mean(rating - mu - b_i - b_u - b_mf - b_ma))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_my) %>%
  pull(pred)

rmse_ma_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_ma_step_temp <- data.frame(Stepwise = "+ Movie Year", 
                                RMSE = rmse_ma_step_temp,
                                Difference = rmse_ma_step[1,2] - rmse_ma_step_temp)

rmse_ma_step <- rbind(rmse_ma_step, rmse_ma_step_temp)
rmse_ma_step

#Plot Figure 11. Baseline Prediction + 3 variables
rmse_ma_step <- rmse_ma_step [order(-rmse_ma_step $RMSE),]

ggplot(rmse_ma_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  plot_theme

################################################################################
#Stepwise modelling (baseline + movie frequency + movie age + user frequency)
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_uf) %>%
  pull(pred)

rmse_my_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_my_step <- data.frame(Stepwise = "Baseline + ... + User Frequency", 
                           RMSE = rmse_my_step_temp,
                           Difference = 0)

#Add: Movie year
movieyear_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  group_by(movieYear) %>% 
  summarize(b_my = mean(rating - mu - b_i - b_u - b_mf - b_ma - b_uf))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_uf + b_my) %>%
  pull(pred)

rmse_my_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_my_step_temp <- data.frame(Stepwise = "+ Movie Year", 
                                RMSE = rmse_my_step_temp,
                                Difference = rmse_my_step[1,2] - rmse_my_step_temp)

rmse_my_step <- rbind(rmse_my_step, rmse_my_step_temp)

#Add: Genre Effect
genres_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_mf - b_ma - b_uf))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_uf + b_g) %>%
  pull(pred)

rmse_my_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_my_step_temp <- data.frame(Stepwise = "+ Genres", 
                                RMSE = rmse_my_step_temp,
                                Difference = rmse_my_step[1,2] - rmse_my_step_temp)

rmse_my_step <- rbind(rmse_my_step, rmse_my_step_temp)

#Plot Figure 12. Baseline Prediction + 4 variables
rmse_my_step <- rmse_my_step [order(-rmse_my_step $RMSE),]

ggplot(rmse_my_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  plot_theme

################################################################################
#Stepwise modelling (baseline + movie frequency + movie age + user frequency + movie year)

#Predict (mu + b_i + b_u + b_mf + b_ma + b_my + b_uf)
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_uf + b_my) %>%
  pull(pred)

rmse_uf_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_uf_step <- data.frame(Stepwise = "Baseline + ... + User Frequency", 
                           RMSE = rmse_uf_step_temp,
                           Difference = 0)

#Add: Genre
genres_avgs <- trainset %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_mf - b_ma - b_uf - b_my))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_uf + b_my + b_g) %>%
  pull(pred)

rmse_fullmodel <- RMSE(predicted_ratings, validation$rating)
rmse_uf_step_temp <- RMSE(predicted_ratings, validation$rating) 
rmse_uf_step_temp <- data.frame(Stepwise = "+ Genres", 
                                RMSE = rmse_uf_step_temp,
                                Difference = rmse_uf_step[1,2] - rmse_uf_step_temp)

rmse_uf_step <- rbind(rmse_uf_step, rmse_uf_step_temp)

#Plot Figure 13. Baseline Prediction + 5 variables
rmse_uf_step <- rmse_uf_step [order(-rmse_uf_step $RMSE),]

ggplot(rmse_uf_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  plot_theme

################################################################################
#Final hold-out test of the complete model

#Create predictions for the Final Model against test set
mu <- mean(train$rating)
movie_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + lambda_bi))
user_avgs <- train %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n() + lambda_bu))
moviefreq_avgs <- train %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(movieFreq) %>% 
  summarize(b_mf = mean(rating - mu - b_i - b_u))
movieage_avgs <- train %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  group_by(movieAge) %>% 
  summarize(b_ma = mean(rating - mu - b_i - b_u - b_mf))
userfreq_avgs <- train %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu - b_i - b_u - b_mf - b_ma))
movieyear_avgs <- train %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  group_by(movieYear) %>% 
  summarize(b_my = mean(rating - mu - b_i - b_u - b_mf - b_ma - b_uf))
genres_avgs <- train %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_mf - b_ma - b_uf - b_my))
predrating_benchmark <- test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_uf + b_my + b_g) %>%
  pull(pred)

#Test Final Model and get the value of RMSE
rmse_final <- RMSE(predrating_benchmark, test$rating) 
rmse_final #0.8632337

rm(lambda_bi, lambda_bu, predrating_benchmark, trainset, validation, genres_avgs,
   movie_avgs, movieage_avgs, moviefreq_avgs, movieyear_avgs, user_avgs, userfreq_avgs, 
   genres_avgs, movie_avgs, movieage_avgs, moviefreq_avgs, movieyear_avgs,
   rmse_ma_step, rmse_ma_step_temp, rmse_mf_step, rmse_mf_step_temp,
   rmse_my_step, rmse_my_step_temp, rmse_uf_step, rmse_uf_step_temp,
   rmse_user_step, rmse_user_step_temp, rmse_fullmodel, user_avgs, userfreq_avgs,
   predicted_ratings, mu)

################################################################################
#Matrix Factorization with Gradient Descent using Recosystem
ratings_train <- train[c(1:3)]
ratings_test <- test[c(1:3)]

write.table(ratings_train, file = "trainset.txt", sep = " ", row.names = FALSE, col.names = FALSE)
write.table(ratings_test, file = "testset.txt", sep = " ", row.names = FALSE, col.names = FALSE)

#This function simply returns an object of class "RecoSys" that can be used to construct recommender model and conduct prediction.
r = Reco()

#Tune model parameters
opts <- r$tune("trainset.txt", opts = list(dim = 30, 
                                           lrate = 0.1,
                                           costp_l1 = 0,
                                           costq_l1 = 0,
                                           nfold = 5,
                                           niter = 20,
                                           nthread = 4))

#This method is a member function of class "RecoSys" that trains a recommender model.
r$train("trainset.txt", opts = c(opts$min, nthread = 4, niter = 500, verbose = TRUE))

outfile = tempfile()

#This method is a member function of class "RecoSys" that predicts unknown entries in the ratingmatrix.
r$predict("testset.txt", outfile)

rating_real <- read.table("testset.txt", header = FALSE, sep = " ")$V3
predrating_mf <- scan(outfile)

rmse_mf <- RMSE(predrating_mf, test$rating) 
rmse_mf #0.7924707

###############################################################################
#Show prediction for a specific user
###############################################################################
#Check profile of users in terms of their frequency of ratings in the dataset
user_profile <- movielens %>% count(userId)

##Start here for predictions
#Set User ID
current_user <- 2 #user must have atleast 1 rating data

#set N, number of movie recommendation
n_recom <- 20

#Create dataset of user's rating
user_ratings <- as.data.frame(movielens) %>%
  filter(movielens$userId == current_user)

###############################################################################
#Show top ratings of the user

user_ratings %>% 
  arrange(-rating) %>%
  slice_head(n = n_recom) %>%
  summarize('User ID' = userId,
            'Movie ID' = movieId,
            'User Rating' = rating,
            'Title' = title,
            'Year Released' = movieYear,
            'Genre' = genres) %>%
  data.table()

################################################################################
#Recommendation: Show top predicted ratings
#Run prediction code
userId <- rep(c(current_user), each = length(movieId)) #userID vector with the same length as movieID
pred_user <- data.frame(userId, movieId) #create dataframe for a userID on all movieID
write.table(pred_user, file = "testset.txt", sep = " ", row.names = FALSE, col.names = FALSE)
r$predict("testset.txt", outfile)
rating_real <- read.table("testset.txt", header = FALSE, sep = " ")$V3
predrating_mf <- scan(outfile)
pred_user <- cbind(pred_user, predrating_mf)

#Slice prediction (for efficiency)
pred_user <- pred_user %>%
  arrange(-predrating_mf) %>%
  top_n(n_recom, wt = predrating_mf)

#Show recommendation in a data table
pred_user %>% 
  arrange(-predrating_mf) %>% 
  left_join(select(test, title, movieYear, genres, movieId), by = "movieId") %>%
  unique() %>%
  summarize('Movie ID' = movieId,
            'Title' = title,
            'Year Released' = movieYear,
            'Genre' = genres) %>%
  data.table()

################################################################################
#Alternative recommendation: Exclude movies already rated by the user
#Run prediction code
pred_user <- data.frame(userId, movieId)
write.table(pred_user, file = "testset.txt", sep = " ", row.names = FALSE, col.names = FALSE)
r$predict("testset.txt", outfile)
rating_real <- read.table("testset.txt", header = FALSE, sep = " ")$V3
predrating_mf <- scan(outfile)
pred_user <- cbind(pred_user, predrating_mf)

#Remove movies already rated by the user
pred_user <- pred_user[! pred_user$movieId %in% user_ratings$movieId, ]

pred_user <- pred_user %>%
  arrange(-predrating_mf) %>%
  top_n(n_recom, wt = predrating_mf)

pred_user %>% 
  arrange(-predrating_mf) %>% 
  left_join(select(test, title, movieYear, genres, movieId), by = "movieId") %>%
  unique() %>%
  summarize('Movie ID' = movieId,
            'Title' = title,
            'Year Released' = movieYear,
            'Genre' = genres) %>%
  data.table()

################################################################################
#Alternative recommendation: Random Top Rated Movies
movielens %>%
  group_by(movieId) %>%
  summarize(m = mean(rating)) %>%
  arrange(-m) %>%
  top_n(n_recom, wt = m) %>%
  sample_n(n_recom) %>%
  left_join(select(train, title, movieYear, genres, movieId), by = "movieId") %>%
  select(-m) %>%
  unique() %>%
  data.table()

################################################################################
#Alternative recommendation: Random Top Rated Movies based on Genre (Comedy)
movielens %>% 
  filter(grepl('Comedy', genres)) %>%
  group_by(movieId) %>%
  summarize(m = mean(rating)) %>%
  arrange(-m) %>%
  top_n(n_recom, wt = m) %>%
  sample_n(n_recom) %>%
  left_join(select(train, title, movieYear, genres, movieId), by = "movieId") %>%
  select(-m) %>%
  unique() %>%
  data.table()
  