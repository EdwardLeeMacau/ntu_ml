# Homework 4

## Problem 1

$$
\begin{align*}
\varphi'(s) &= \theta(s) + s \cdot \theta'(s) \\
            &= \theta(s) + s \cdot \theta(s) \cdot (1 - \theta(s))
\end{align*}
$$

## Problem 2

Find $v_1, \dots v_5$ by iterative approach.

$$
\begin{align*}
v_1 &= [0.5       , 0.16666667, 0.33333333] \\
v_2 &= [0.33333333, 0.16666667, 0.5       ] \\
v_3 &= [0.41666667, 0.25      , 0.33333333] \\
v_4 &= [0.41666667, 0.16666667, 0.41666667] \\
v_5 &= [0.375     , 0.20833333, 0.41666667] \\
...\\
v^* &= [0.4, 0.2, 0.4]
\end{align*}
$$

Solve $v^* = Pv^*$ by finding the null space of $I - P$

$$
\begin{align*}
        v^* &= Pv^* \\
(I - P) v^* &= 0
\end{align*}
$$

It has the basis $[2/3, 1/3, 2/3]$, after normalized ($||v||_1 = 1$)

$$
v^* = [0.4, 0.2, 0.4]
$$

## Problem 3

