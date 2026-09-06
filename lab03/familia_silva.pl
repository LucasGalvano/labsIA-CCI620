% ============================================================
% ENTREGA - FAMÍLIA SILVA
% ============================================================

% ------------------------------------------------------------
% Fatos base: sexo dos indivíduos
% ------------------------------------------------------------
homem(jose).
homem(joao).
homem(paulo).
homem(carlos).

mulher(maria).
mulher(ana).
mulher(helena).
mulher(joana).

% ------------------------------------------------------------
% b) progenitor(X, Y) -> X é progenitor de Y
% ------------------------------------------------------------
progenitor(jose, joao).
progenitor(jose, ana).
progenitor(maria, joao).
progenitor(maria, ana).

progenitor(ana, helena).
progenitor(ana, joana).

progenitor(joao, paulo).

progenitor(helena, carlos).
progenitor(paulo, carlos).

% ------------------------------------------------------------
% c) Relações de parentesco: pai, mãe, irmão, irmã
% ------------------------------------------------------------
pai(X, Y) :- progenitor(X, Y), homem(X).
mae(X, Y) :- progenitor(X, Y), mulher(X).

% X e Y são irmãos se compartilham pelo menos um progenitor
irmao(X, Y) :-
    progenitor(P, X),
    progenitor(P, Y),
    homem(X),
    X \= Y.

irma(X, Y) :-
    progenitor(P, X),
    progenitor(P, Y),
    mulher(X),
    X \= Y.

% ------------------------------------------------------------
% Predicados auxiliares (necessários para as perguntas d.3, d.4, d.5)
% ------------------------------------------------------------

% ascendente(X, Y) -> X é ascendente de Y (pai/mãe, avô/avó, etc.)
ascendente(X, Y) :- progenitor(X, Y).
ascendente(X, Y) :- progenitor(X, Z), ascendente(Z, Y).

% tio/tia: irmão/irmã de um dos progenitores de Y
tio(X, Y) :- progenitor(P, Y), irmao(X, P).
tia(X, Y)  :- progenitor(P, Y), irma(X, P).

% sobrinho/sobrinha: relação inversa de tio/tia
sobrinho(X, Y) :- tio(Y, X), homem(X).
sobrinho(X, Y) :- tia(Y, X), homem(X).
sobrinha(X, Y) :- tio(Y, X), mulher(X).
sobrinha(X, Y) :- tia(Y, X), mulher(X).

% primo/prima: filhos de tios/tias
primo(X, Y) :-
    progenitor(P, X),
    (irmao(P, Q) ; irma(P, Q)),
    progenitor(Q, Y),
    homem(Y).

prima(X, Y) :-
    progenitor(P, X),
    (irmao(P, Q) ; irma(P, Q)),
    progenitor(Q, Y),
    mulher(Y).

% ============================================================
% d) CONSULTAS
% ============================================================

% 1. O João é filho do José?
%    ?- pai(jose, joao).
%    -> true.

% 2. Quem são os filhos da Maria?
%    ?- findall(X, progenitor(maria, X), Filhos).
%    -> Filhos = [joao, ana].

% 3. Quem são os primos do Paulo?
%    ?- setof(X, (primo(paulo, X) ; prima(paulo, X)), Primos).
%    -> Primos = [helena, joana].

% 4. Quais são os sobrinhos/sobrinhas de cada Tio na família Silva?
%    ?- setof(Tio-Sobrinho, sobrinho(Sobrinho, Tio), R1).
%    ?- setof(Tio-Sobrinha, sobrinha(Sobrinha, Tio), R2).
%    (na família Silva, só há tia: Ana é tia de Paulo; e tio: João é tio de Helena/Joana)

% 5. Quem são os ascendentes do Carlos?
%    ?- setof(X, ascendente(X, carlos), Ascendentes).
%    -> Ascendentes = [ana, helena, joao, jose, maria, paulo].

% 6. A Helena tem irmãos? E irmãs?
%    ?- setof(X, irmao(X, helena), Irmaos).      -> false (nenhum irmão homem)
%    ?- setof(X, irma(X, helena), Irmas).        -> Irmas = [joana].
