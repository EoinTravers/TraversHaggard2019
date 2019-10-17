## functions.R

colours = c('darkgreen', 'red', 'black')
label.order = c('Easy', 'Difficult', 'Guess')

mean.split = function(x) factor((x - mean(x) > 0))
median.split = function(x) factor(sign(x - median(x)))

label.cut = function(x, bins){
  bx = cut(x, bins)
  levels(bx) = 1:length(levels(bx))
  bx
}
qcut = function(x, bins){
  bx = Hmisc::cut2(x, g=bins)
  levels(bx) = 1:length(levels(bx))
  bx
}

z = function(x){
  (x - mean(x)) / sd(x)
}


nlopt <- function(par, fn, lower, upper, control) {
  .nloptr <<- res <- nloptr(par, fn, lb = lower, ub = upper, 
                            opts = list(algorithm = "NLOPT_LN_BOBYQA", print_level = 1,
                                        maxeval = 1000, xtol_abs = 1e-6, ftol_abs = 1e-6))
  list(par = res$solution,
       fval = res$objective,
       conv = if (res$status > 0) 0 else res$status,
       message = res$message
  )
}

load.erp.data = function() {
  data = read_csv('data/long_response_vmax.csv')
  ## Baseline EEG
  .baseline = data %>%
    filter(time > -2.1, time < -2) %>%
    group_by(participant, trial_nr) %>%
    summarise(bl = mean(ch1))
  data = inner_join(data, .baseline, 
                    by=c('participant', 'trial_nr')) %>%
    mutate(ch1 = ch1 - bl)
  rm(.baseline)
  ## Crop data, and basic labelling
  data = data %>%
    filter(rt < 3, time > -2) %>%
    mutate(action.type = ifelse(visible, ifelse(difficult==0, 'Easy', 'Difficult'), 'Guess'),
           ch1 = ch1 / sd(ch1),
           p.act = predicted_action,
           p.bet = predicted_response,
           action.type = factor(action.type, levels=label.order),
           unique.trial = interaction(trial_nr, participant))
  ## RT bins
  data = data %>%
    group_by(participant) %>%
    mutate(median.rt = median(rt),
           median.p.act = median(p.act),
           median.p.bet = median(p.bet),
           speed.quantile = qcut(rt, 4)) %>%
    ungroup() %>%
    mutate(fast = ifelse(rt < median.rt, 'Fast', 'Slow') %>% factor() %>% relevel(ref='Fast') )
  # with(data, plot(fast, rt, ylab='RT')) ## Double-checking
  return(data)
}

get.standard.trials = function(data){
  standard.trials = filter(data, visible==T)
  standard.trials = mutate(standard.trials,
                           iv = interaction(action.type, fast) %>% droplevels)
  standard.trials$iv %<>% factor(., levels=levels(.)[c(2, 4, 1, 3)])
  standard.trials$action.type %<>% droplevels() %>% relevel(ref='Easy')
  levels(standard.trials$action.type) = c('Easy decision (Strong evidence)', 'Difficult decision (Weak evidence)')
  return(standard.trials)
}

get.uncertain.trials = function(data){
  uncertain.trials = filter(data, action.type %in% c('Difficult', 'Guess'))
  uncertain.trials = mutate(uncertain.trials,
                            iv = interaction(action.type, fast) %>% droplevels,
                            is.guess = ifelse(action.type=='Guess', 1, 0))
  uncertain.trials$iv %<>% factor(., levels=levels(.)[c(2, 4, 1, 3)])
  uncertain.trials$action.type %<>%  factor() %>% droplevels() %>% relevel(ref='Guess')
  levels(uncertain.trials$action.type) = c('Guess (Endogenous evidence)', 'Difficult decision (Exogenous evidence)')
  return(uncertain.trials)
}




get.marginal.fx = function(data, sims, var.name=NA, spline.df=5) {
  t = unique(data$time)
  reference.data = data.frame(bs(t, df=spline.df))
  if(is.na(var.name) | var.name=='intercept'){
    coef.name0 = sprintf('bs(time, df = df)1')
    coef.name1 = sprintf('bs(time, df = df)%i', spline.df)
  } else{
    coef.name0 = sprintf('bs(time, df = df)1:%s', var.name)
    coef.name1 = sprintf('bs(time, df = df)%i:%s', spline.df, var.name)
  }
  print(c(coef.name0, coef.name1))
  i0 = which.max(colnames(sims) == coef.name0)
  i1 = which.max(colnames(sims) == coef.name1)
  X = as.matrix(reference.data)
  B = sims[,i0:i1] %>% t()
  Y = matrix(NA, nrow(X), ncol(B))
  for(i in 1:nrow(Y)) {
    for(j in 1:ncol(Y)){
      Y[i, j] = sum(X[i,] * B[,j])
    }
  }
  df = data.frame(t, Y) %>% gather(samp, val, -t)
  df
}



get.predictions = function(data, sims, model) {
  full.mm = model.matrix(model)
  ref.df = data.frame(full.mm, fast=data$fast, action.type=data$action.type, time=data$time)
  ref.df = ref.df %>%
    group_by(time, fast, action.type) %>%
    summarise_all(mean) %>%
    ungroup()
  output.df = ref.df %>% select(time, fast, action.type)
  X = ref.df %>% select(-time, -fast, -action.type) %>% as.matrix()
  B = t(sims)
  # dim(X)
  # dim(B)
  Y = matrix(NA, nrow(X), ncol(B))
  dim(Y)
  for(i in 1:nrow(Y)) {
    for(j in 1:ncol(Y)){
      Y[i, j] = sum(X[i,] * B[,j])
    }
  }
  df = data.frame(output.df, Y) %>% gather(samp, val, -time, -fast, -action.type)
  return(df)
}

get.marginal.splines = function(data, sims, var.name=NA, spline.df=5) {
  t = unique(data$time)
  reference.data = data.frame(bs(t, df=spline.df))
  if(is.na(var.name) | var.name=='intercept'){
    coef.name0 = sprintf('bs(time, df = df)1')
    coef.name1 = sprintf('bs(time, df = df)%i', spline.df)
  } else{
    coef.name0 = sprintf('bs(time, df = df)1:%s', var.name)
    coef.name1 = sprintf('bs(time, df = df)%i:%s', spline.df, var.name)
  }
  print(c(coef.name0, coef.name1))
  i0 = which.max(colnames(sims) == coef.name0)
  i1 = which.max(colnames(sims) == coef.name1)
  X = as.matrix(reference.data)
  B = sims[,i0:i1] %>% t()
  Y = array(NA, c(nrow(X), ncol(B), spline.df))
  for(i in 1:nrow(Y)) {
    for(j in 1:ncol(Y)){
      Y[i, j, ] = X[i,] * B[,j]
    }
  }
  res = map_df(1:spline.df, function(b){
    data.frame(t, Y[,,b])  %>% gather(samp, val, -t) %>% mutate(basis=b)
  }) %>% mutate(basis=factor(basis))
  return(res)
}
