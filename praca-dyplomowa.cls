% autor: Paweł Mleczko <pml@amu.edu.pl> Pliki praca-dyplomowa.cls oraz
% praca.tex są objęte licencją Creative Commons BY-NC-SA szczegóły
% http://creativecommons.org/licenses/by-nc-sa/3.0/pl/legalcode

%% Klasa przygotowana do pisania prac dyplomowych z matematyki,
%% na Wydziale Matematyki i Informatyki UAM w Pozanniu 
%% copyright Paweł Mleczko 2016
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{praca-dyplomowa}[2015/3/03 Bachelor's degree thesis class file]
\LoadClass[11pt,oneside]{mwart}
%% page dimension
\baselineskip=15pt
\RequirePackage[a4paper,
textwidth=135mm,lines=42,
vmarginratio=1:1,hmarginratio=1:1,bindingoffset=0mm,%-6.5mm
headheight=9pt,headsep=25pt,footskip=25pt]{geometry}
%% headers, footers
\RequirePackage{fancyhdr}
            \pagestyle{fancy}
            \lhead[\thepage]{{\fontsize{9pt}{9pt}\selectfont\emph{\leftmark}}}
            \chead[]{}
            \rhead[{\fontsize{9pt}{9pt}\selectfont\emph{\rightmark}}]{\thepage}
            \lfoot[]{}
            \cfoot[]{}
            \rfoot[]{}
            \renewcommand{\headrulewidth}{0pt}
            \renewcommand{\footrulewidth}{0pt}
            \fancypagestyle{titlepage}{%
              \lhead[]{} \chead[]{} \rhead[]{}
              \rfoot[]{} \cfoot[]{\thepage}
              \lfoot[]{}
              \renewcommand{\headrulewidth}{0pt}
              \renewcommand{\footrulewidth}{0pt}}
%% math packages
\RequirePackage{amsmath}
\RequirePackage{amssymb}
%% tikz
\RequirePackage{tikz,pgfplots}
%% miscellanea
\RequirePackage{lipsum}
\RequirePackage{xparse}
\RequirePackage{etoolbox}
\RequirePackage{url}
\urlstyle{same}
\let\oldurl\url
\renewcommand{\url}[1]{<\oldurl{#1}>}
%% variable
\ExplSyntaxOn
\def\title#1{\def\Title{#1}}
% \def\entitle#1{\def\Entitle{#1}}
\def\author#1{\def\Author{#1}}
\def\album#1{\def\Album{#1}}
\def\year#1{\def\Year{#1}}
\def\promotor#1{\def\Promotor{#1}}
\def\type#1{\def\Type{#1}}
\def\course#1{\def\Course{#1}}
\def\speciality#1{\def\Speciality{#1}}
\def\date#1{\def\Date{#1}}
\ExplSyntaxOff
%% table of contents
\newwrite\Tableofcontents
\def\Spis#1#2#3{%
\noindent\llap{\hbox to 30pt{\hfill\textbf{#1}~~}}\textbf{#2\nobreak\hskip 4pt\nobreak...\nobreak\hskip 4pt\nobreak{}#3}\par\vskip 6pt}
\def\spis#1#2{%
\noindent#1\nobreak\hskip 4pt\nobreak...\nobreak\hskip 4pt\nobreak{}#2\par\vskip 6pt}
\def\tableofcontents{%
  \clearpage\mbox{ }
\vskip 2.5\baselineskip
  \thispagestyle{empty}
\vbox to 10\baselineskip{%
    \bgroup\fontsize{18pt}{24pt}\selectfont\centering% 
    \textbf{Spis tre{\'s}ci}\par
    \egroup%
  }
\bgroup\fontsize{11pt}{15pt}\selectfont\raggedright\leftskip=30pt
\IfFileExists{./\jobname.tf}{\input{\jobname.tf}}{\relax}\par
\egroup%
\immediate\openout\Tableofcontents=\jobname.tf
}
%% sections
\ExplSyntaxOn
\def\dekoracja{\vrule width 40pt height .75pt depth 0pt\quad \vrule
  width 1.5pt height 1.5pt depth 0pt \quad\vrule width 40pt height
  .75pt depth 0pt\par \vskip .75\baselineskip}

\RenewDocumentCommand\section{sm}{%
  \clearpage\mbox{ }
  \vskip 2\baselineskip\thispagestyle{titlepage}
  \IfBooleanTF#1%
  {%
    \vbox to 10\baselineskip{%
      \bgroup\fontsize{16pt}{22pt}\selectfont\centering% 
      \vskip 1.5\baselineskip
      \noindent\textbf{#2}\par
      % \dekoracja
      \egroup%
      \immediate\write\Tableofcontents{%
        \unexpanded{\Spis}{}{\unexpanded{#2}}{\thepage}}%
  }\markboth{#2}{#2}}% with a star
  {%
    \refstepcounter{section}\setcounter{Counter}{0}%
    \vbox to 10\baselineskip{%      
      \bgroup\fontsize{16pt}{22pt}\selectfont\centering%
      \textbf{Chapter~\thesection}
      \vskip .5\baselineskip
      \textbf{#2}\par
      % \dekoracja
      \egroup
      \immediate\write\Tableofcontents{%
        \unexpanded{\Spis}{\thesection.}{\unexpanded{#2}}{\thepage}}% 
}\markboth{Chapter~\thesection.~#2}{Chapter~\thesection.~#2}%
}% without a star
} 

\newcounter{Counter} \setcounter{Counter}{0}
\renewcommand{\theCounter}{\thesection.\arabic{Counter}}

\RenewDocumentCommand\subsection{sm}{%
\addvspace{2\baselineskip plus .5\baselineskip}
%
  \IfBooleanTF#1%
    {%
       \bgroup\fontsize{12pt}{16pt}\selectfont\centering% 
       \noindent\textbf{#2}\par\write\Tableofcontents{%
        \unexpanded{\spis}{\unexpanded{#2}}{\thepage}}%
       \egroup%
    }% with a star
    {%
      \bgroup\fontsize{12pt}{16pt}\selectfont%
      \noindent\textbf{#2}\par\write\Tableofcontents{%
        \unexpanded{\spis}{\unexpanded{#2}}{\thepage}}%
      \egroup% 
}% without a star
\addvspace{1\baselineskip plus .5\baselineskip}}
\ExplSyntaxOff
%% enumerations etc
\RequirePackage{enumitem}
\setlist[enumerate,1]{label={\textup{\arabic*)}}}
\setlist[enumerate,2]{label=\roman*)}
\setlist[itemize,1]{label=--,leftmargin=0pt,itemindent=1.5em,topsep=0pt}
%% abstracts
\DeclareDocumentEnvironment{streszczenie}{}%
{%
  \clearpage\thispagestyle{empty}
  \noindent\textbf{\Title}\par
  \vskip 1\baselineskip
    \bgroup\itshape
  \noindent \textbf{\textup{Streszczenie}.}
}%
{%
  \egroup\par
}
\DeclareDocumentEnvironment{abstract}{}%
{%
\vskip 3\baselineskip
  \noindent\textbf{\Title}\par
  \vskip 1\baselineskip
    \bgroup\itshape
  \noindent \textbf{\textup{Abstract}.}
}%
{%
  \egroup\par
}
\DeclareDocumentEnvironment{acknowlegements}{}%
{%
\vskip 3\baselineskip
  \noindent\textbf{\Title}\par
  \vskip 1\baselineskip
    \bgroup\itshape
  \noindent \textbf{\textup{Acknowlegements}.}
  \newline
}%
{%
  \egroup\par
}
%% theorems etc
\ExplSyntaxOn
\DeclareDocumentCommand{\italicsStyle}{}{%
\def\Style{\itshape}}
\DeclareDocumentCommand{\plainStyle}{}{%
\def\Style{}}
\DeclareDocumentCommand\newtheorem{mm}{%
\csedef{#1style}{\expandonce\Style}
\DeclareDocumentEnvironment{#1}{o}%
{%
\par\refstepcounter{Counter}\addvspace{.5\baselineskip plus
.5\baselineskip}
\IfNoValueTF{##1}{%
    \bgroup\csuse{#1style}%
    \noindent \textbf{\textup{\theCounter.~#2}.}
}{%
    \bgroup\csuse{#1style}%
    \noindent \textbf{\textup{\theCounter.~#2~(##1)}.}
}%
}%
{%
    \par
\IfNoValueTF{#1}{\egroup}{\egroup}\par
\addvspace{.5\baselineskip plus .5\baselineskip}
}}
%
\def\qed{\(\square\)}
\DeclareDocumentEnvironment{proof}{}%
{%
\noindent\emph{Proof}.}
{\nobreak\hfill\qed\par
\addvspace{1\baselineskip}}
\ExplSyntaxOff
%% typography
\RequirePackage{microtype}
\newcommand{\polishendash}{\ifhmode\unskip\fi\discretionary{--}{--}{--}\nolinebreak}
%% Titlepage
\def\Titlepage{%
\thispagestyle{empty}
\bgroup\fontsize{14pt}{18pt}\selectfont\centering
Uniwersytet im. Adama Mickiewicza w Poznaniu\\
Wydzia{\l} Matematyki i Informatyki

\vskip 5\baselineskip

\bgroup\fontsize{18pt}{24pt}\selectfont\centering\bfseries
\Title\par
\egroup

\vskip 1\baselineskip

% Drugi tytuł odkomentuj tutaj
% \bgroup\fontsize{14pt}{20pt}\selectfont\centering\bfseries
% \Title\par
% \egroup

% \vskip 2\baselineskip

\bgroup\fontsize{14pt}{14pt}\selectfont\centering
\Author\par
\egroup

\vskip 1\baselineskip

\bgroup\fontsize{11pt}{15pt}\selectfont\centering
nr albumu \Album\par
\egroup

\vfill

\hfill\begin{minipage}{.5\textwidth}%
\bgroup\fontsize{11pt}{15pt}\selectfont\raggedright
praca \Type{}\par
kierunek: \Course\par
specjalno\'s\'c: \Speciality\par
promotor: \Promotor\par
\egroup
\end{minipage}

\vskip 6\baselineskip

Pozna\'n{} \Year\par
\egroup\pagebreak}
%%
\def\gender#1{#1}
\def\male{\maletrue}
\def\female{\malefalse}
\newif\ifmale
\def\Sfirst#1{\def\@stat{#1}}
\def\Ssecond#1{\def\@Stat{#1}}
\def\Statement{%
\thispagestyle{empty}
\ifmale
\hfill Pozna{\'n}, dnia \Date~r.
        
\vfill

{\centering\bfseries O{\'s}wiadczenie\par}

\vskip 3\baselineskip

\noindent Ja, ni{\.z}ej podpisany \textbf{\Author{}} student
Wydzia{\l}u Matematyki i~Informatyki Uniwersytetu im. Adama
Mickiewicza w Poznaniu o{\'s}wiadczam, \.ze przedk{\l}adan\k{a}
prac\k{e} dyplomow\k{a} pt: \textbf{\Title} napisa{\l}em
samodzielnie. Oznacza to, \.z{}e przy pisaniu pracy, poza
niezb\k{e}dnymi konsultacjami, nie korzysta{\l}em z~pomocy innych
os\'o{}b, a~w~szczeg\'o{}lno{\'s}ci nie zleca{\l}em opracowania rozprawy
lub jej cz\k{e}\'s{}ci innym osobom, ani nie odpisywa{\l}em tej
rozprawy lub jej cz\k{e}\'sci od innych os\'o{}b.

O\'s{}wiadczam r\'o{}wnie\.z, \.z{}e egzemplarz pracy dyplomowej
w~wersji drukowanej jest ca{\l}kowicie zgodny z egzemplarzem pracy
dyplomowej w wersji elektronicznej.

\else

\hfill Pozna{\'n}, dnia \Date~r.
        
\vfill

{\centering\bfseries O{\'s}wiadczenie\par}

\vskip 3\baselineskip

\noindent Ja, ni{\.z}ej podpisana \textbf{\Author{}} studentka
Wydzia{\l}u Matematyki i~Informatyki Uniwersytetu im. Adama
Mickiewicza w Poznaniu o{\'s}wiadczam, \.ze przedk{\l}adan\k{a}
prac\k{e} dyplomow\k{a} pt: \textbf{\Title} napisa{\l}am
samodzielnie. Oznacza to, \.z{}e przy pisaniu pracy, poza
niezb\k{e}dnymi konsultacjami, nie korzysta{\l}am z~pomocy innych
os\'o{}b, a~w~szczeg\'o{}lno{\'s}ci nie zleca{\l}am opracowania rozprawy
lub jej cz\k{e}\'s{}ci innym osobom, ani nie odpisywa{\l}am tej
rozprawy lub jej cz\k{e}\'sci od innych os\'o{}b.

O\'s{}wiadczam r\'o{}wnie\.z, \.z{}e egzemplarz pracy dyplomowej
w~wersji drukowanej jest ca{\l}kowicie zgodny z egzemplarzem pracy
dyplomowej w wersji elektronicznej.

\fi

Jednocze{\'s}nie przyjmuj\k{e} do wiadomo{\'s}ci, \.ze przypisanie
sobie, w pracy dyplomowej, autorstwa istotnego fragmentu lub innych
element\'ow cudzego utworu lub ustalenia naukowego stanowi
podstaw\k{e} stwierdzenia niewa\.zno{\'s}ci post\k{e}powania w sprawie
nadania tytu{\l}u zawodowego.

\vskip 1\baselineskip

\noindent [\@stat]\,* - wyra\.zam zgod\k{e} na
udost\k{e}pnianie mojej pracy w czytelni Archiwum UAM

\vskip .5\baselineskip

\noindent [\@Stat]\,* - wyra\.zam zgod\k{e} na udost\k{e}pnianie mojej pracy w zakresie
koniecznym do ochrony mojego prawa do autorstwa lub praw os\'ob trzecich

\vskip 5\baselineskip

\hfill\begin{minipage}{.5\textwidth}\fontsize{9pt}{9pt}\selectfont\centering
(czytelny podpis studenta)\par
\end{minipage}

\vfill

\bgroup\fontsize{9pt}{11pt}\selectfont
\noindent *\,Nale\.zy wpisa{\'c} TAK w przypadku wyra{\.z}enia zgody na udost\k{e}pnianie pracy
w czytelni Archiwum UAM, NIE w przypadku braku zgody. Niewype{\l}nienie
pola oznacza brak zgody na udost\k{e}pnianie pracy.\par
\egroup\clearpage}


%% before start
\AtBeginDocument{\baselineskip=15pt\def\le{\leq}
\def\ge{\geq}}
\AtEndDocument{\immediate\closeout\Tableofcontents}
\endinput