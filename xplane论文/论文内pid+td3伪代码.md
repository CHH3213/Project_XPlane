```
The pseudocode of our TD3 is shown in Algorithm \ref{algo_TD3}.
\begin{algorithm}
\label{algo_TD3}
    \caption{Modified TD3 algorithm}%算法名字
    \LinesNumbered %要求显示行号
    \KwIn{initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, empty replay buffer $\mathcal{D}$, terminated pre-train condition $E_{threshold}$}
    % \KwOut{output result}%输出
    Set target parameters equal to main parameters $\theta_{targ}\leftarrow \theta$, $\phi_{targ,1}\leftarrow \phi_1$, $\phi_{targ,2}\leftarrow \phi_2$\;
    Set Pre-train PID policy with a greedy exploration\;
    \For{episode=1 to M}{
        Receive initial observation state $s_1$\;
        \If{episode $< E_{threshold}$}{
            Initialize a pre-train a guidance policy using PID\;
        }
        \Else{
            Initialize a random process $\mathcal{N}$ for action exploration\;
            }
        
        \For{t=1 to T}{
            Select action $a_t$ according to the current policy and exploration noise or the guidance policy\;
            
            Execute action $a_t$ and observe reward $r_t$ and observe new state $s_{t+1}$ \;
            Normalize the state $s_{t+1}$\;
            Get $r_t^{rs}$ using 
            $ r_t^{rs}=T(s_{t+1})+F(s_t,a_t,s_{t+1})$
            
            Store transition $(s_t, a_t, r_t^{rs}, s_{t+1})$ in $\mathcal{D}$\;
            
            Sample a mini-batch of transitions,  \quad$\mathcal{B}=\{(s,a,r^{rs},s',d)\}$  from $\mathcal{D}$\;
            
            Compute target actions $a'(s')=$
             \quad$clip(\pi_{\theta_{targ}}(s')+clip(\epsilon,-c,c),a_{Low},a_{High})$\;
            
            Compute targets $y(r^{rs},s',d) =$
             \quad$ r^{rs}+\gamma(1-d)\min_{i=1,2}Q_{\phi_{i,targ}}(s',a')$\;
            
            Update Q-functions using
                 $\nabla_{\phi_{i}}\frac{1}{\mathcal{B}}\sum_{(s,a,s',r^{rs},d)\in \mathcal{B}}{(Q_{\phi_{i}}(s,a)-y(r^{rs},s',d))^2}$
            \If{t mod $p_{delay}$=0}{
                Update policy using
                  \quad$\nabla_{\theta}\frac{1}{\mathcal{B}}\sum_{s\in \mathcal{B}}{Q_{\phi_{1}}(s,\pi_{\theta}(s))}$\;
                  
                Update the target network:\
             \quad\quad$\phi_{targ,i}\leftarrow\tau\phi_{targ,i}+(1-\tau)\phi_{i}$\
             \quad\quad$\theta_{targ}\leftarrow\tau\theta_{targ}+(1-\tau)\theta$
                }
        }
    }
\end{algorithm}
```

纯TD3	伪代码

```
\begin{algorithm}
\label{algo_TD3}
    \caption{TD3-based upset recovery algorithm}%算法名字
    \LinesNumbered %要求显示行号
    \KwIn{initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, empty replay buffer $\mathcal{D}$}
    % \KwOut{output result}%输出
    Set target parameters equal to main parameters $\theta_{targ}\leftarrow \theta$, $\phi_{targ,1}\leftarrow \phi_1$, $\phi_{targ,2}\leftarrow \phi_2$\;
    \For{episode=1 to M}{
        Receive initial observation state $s_1$\;
        Initialize a random process $\mathcal{N}$ for action exploration\;
        \For{t=1 to T}{
            Select action $a_t$ according to the current policy and exploration noise or the guidance policy\;
            
            Execute action $a_t$ and observe reward $r_t$ and observe new state $s_{t+1}$ \;
            Normalize the state $s_{t+1}$\;
            Get $r_t^{rs}$ using 
            $ r_t^{rs}=T(s_{t+1})+F(s_t,a_t,s_{t+1})$
            
            Store transition $(s_t, a_t, r_t^{rs}, s_{t+1})$ in $\mathcal{D}$\;
            
            Sample a mini-batch of transitions,  \quad$\mathcal{B}=\{(s,a,r^{rs},s',d)\}$  from $\mathcal{D}$\;
            
            Compute target actions $a'(s')=$
             \quad$clip(\pi_{\theta_{targ}}(s')+clip(\epsilon,-c,c),a_{Low},a_{High})$\;
            
            Compute targets $y(r^{rs},s',d) =$
             \quad$ r^{rs}+\gamma(1-d)\min_{i=1,2}Q_{\phi_{i,targ}}(s',a')$\;
            
            Update Q-functions using
                 $\nabla_{\phi_{i}}\frac{1}{\mathcal{B}}\sum_{(s,a,s',r^{rs},d)\in \mathcal{B}}{(Q_{\phi_{i}}(s,a)-y(r^{rs},s',d))^2}$
            \If{t mod $p_{delay}$=0}{
                Update policy using
                  \quad$\nabla_{\theta}\frac{1}{\mathcal{B}}\sum_{s\in \mathcal{B}}{Q_{\phi_{1}}(s,\pi_{\theta}(s))}$\;
                  
                Update the target network:\
             \quad\quad$\phi_{targ,i}\leftarrow\tau\phi_{targ,i}+(1-\tau)\phi_{i}$\
             \quad\quad$\theta_{targ}\leftarrow\tau\theta_{targ}+(1-\tau)\theta$
                }
        }
    }
\end{algorithm}

```



3.

```
\begin{algorithm}
\label{algo_TD3}
    \caption{TD3-based upset recovery algorithm for aircraft}%算法名字
    \LinesNumbered %要求显示行号
    \KwIn{initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, empty replay buffer $\mathcal{D}$}
    % , terminated pre-train condition $R_{threshold}$}
    % \KwOut{output result}%输出
    Set target parameters equal to main parameters $\theta_{targ}\leftarrow \theta$, $\phi_{targ,1}\leftarrow \phi_1$, $\phi_{targ,2}\leftarrow \phi_2$\;
    \If{ \textbf{Pre-train} stage}{
        Set Pre-train  aircraft model parameters and environment parameters randomly\;}
    \If{ \textbf{Fine-tuning} stage}{
        Set the pre-train agent as the baseline agent and obtain the guidance policy\;
        Reset aircraft model parameters and environment parameters\;
    }
    % Pre-train a guidance agent using original TD3\;
    % Set the guidance agent as the baseline agent\;
    % Reset aircraft model parameters\;
    \For{episode=1 to M}{
        Receive initial observation state $s_1$\;
        Normalize the initial state $s_1$\;
        \For{t=1 to T}{
            \If{ \textbf{Pre-train} stage}{
                Select and clip action $a$ with exploration noise $\epsilon$\;}
            \If{ \textbf{Fine-tuning} stage}{
               Select and clip action $a$ according to the guidance policy\;
            }
            Convert the action into the aircraft control\;
            Execute action $a$ in aircraft environment, and obtain \textbf{shaped reward} $r$ and new state $s'$\;
            Normalize the state $s'$\;
            % Shape $r$ using $T(s')+F(s,a,s')$\;
            Store transition $(s, a, r, s',d)$ in $\mathcal{D}$\;
            Sample a mini-batch of transitions \quad$\mathcal{B}=\{(s,a,r,s',d)\}$  from $\mathcal{D}$\;
            Compute target actions $a'(s')=$
             \quad$clip(\pi_{\theta_{targ}}(s')+clip(\epsilon,-c,c),a_{Low},a_{High})$\;
            
            Compute targets $y(r,s',d) =$
             \quad$ r+\gamma(1-d)\min_{i=1,2}Q_{\phi_{i,targ}}(s',a')$\;
            
            Update Q-functions using
                 $\nabla_{\phi_{i}}\frac{1}{\mathcal{B}}\sum_{(s,a,s',r,d)\in \mathcal{B}}{(Q_{\phi_{i}}(s,a)-y(r,s',d))^2}$
            \If{t mod $p_{delay}$=0}{
                Update policy using
                  \quad$\nabla_{\theta}\frac{1}{\mathcal{B}}\sum_{s\in \mathcal{B}}{Q_{\phi_{1}}(s,\pi_{\theta}(s))}$\;
                  
                Update the target network:\
             \quad\quad$\phi_{targ,i}\leftarrow\tau\phi_{targ,i}+(1-\tau)\phi_{i}$\
             \quad\quad$\theta_{targ}\leftarrow\tau\theta_{targ}+(1-\tau)\theta$
                }
        }
    }
\end{algorithm}
```

