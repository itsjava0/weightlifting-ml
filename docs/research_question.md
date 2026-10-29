1. Question: For my top 5 exercises, can I predict next-session e1RM more accurately than the "last_value" baseline by incorporating training load, frequency, and recovery variables?
2. Hypothesis: Recent training volume and session frequency will carry signal, because higher cumulative load and more-frequent training should precede performance gains (up to a point)
3. Predictors: weight lifted, reps, volume, days-since-last-session, sessions-in-trailing-7-days, average sleep (if aligned to session date)
4. Target: e1RM on the next session for the same exercise
5. Success criterion: Beat MAE 6.45kg (the baseline) by a statistically meaningful margin (your walk-forward CV will tell you if it's real or noise).


