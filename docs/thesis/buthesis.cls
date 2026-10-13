%% =====================================================================
%%  buthesis.cls  --  Boston University General Thesis Template (LaTeX)
%%
%%  A LaTeX reproduction of the BU "General Thesis Template" (MS Word,
%%  July 2026 version). Page geometry, fonts, spacing, heading styles,
%%  page numbering, TOC / list styles and captions follow the Word file.
%%
%%  Compile with pdfLaTeX (or LuaLaTeX) + BibTeX:
%%      latexmk -pdf thesis.tex
%% =====================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{buthesis}[2026/09/30 BU General Thesis Template (LaTeX port)]

% ---------------------------------------------------------------------
%  Options
%    notes    : show the template's italic instruction notes (default)
%    nonotes  : hide them (use this for the final submission)
% ---------------------------------------------------------------------
\newif\ifbu@shownotes \bu@shownotestrue
\DeclareOption{notes}{\bu@shownotestrue}
\DeclareOption{nonotes}{\bu@shownotesfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

\LoadClass[12pt,letterpaper,oneside]{report}

% ---------------------------------------------------------------------
%  Packages
% ---------------------------------------------------------------------
\RequirePackage[T1]{fontenc}
% Times New Roman equivalent (newtx preferred; mathptmx fallback)
\IfFileExists{newtxtext.sty}
  {\RequirePackage{newtxtext,newtxmath}}
  {\RequirePackage{mathptmx}}
\RequirePackage{etoolbox}
\RequirePackage{setspace}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{titletoc}
\RequirePackage{graphicx}
\RequirePackage{float}                 % [H] placement, as in Word
\RequirePackage{newfloat}              % "Illustration" float type
\RequirePackage{caption}
\RequirePackage{array,tabularx}
\RequirePackage[round]{natbib}

% ---------------------------------------------------------------------
%  Page geometry (Word: top 1.5in, left 1.5in, right 1in, bottom 1in;
%  header 1in from top edge, footer 0.75in from bottom edge)
% ---------------------------------------------------------------------
\RequirePackage[letterpaper,
  top=1.5in, bottom=1in, left=1.5in, right=1in,
  headheight=15pt, headsep=0.33in, footskip=14.5pt]{geometry}

% ---------------------------------------------------------------------
%  Line spacing.  Word "single" for 12pt Times = 13.8pt; "double" = 27.6pt
% ---------------------------------------------------------------------
\newcommand{\businglespacing}{\setstretch{0.9517}}  % 13.8pt baselines
\newcommand{\budoublespacing}{\setstretch{1.9034}}  % 27.6pt baselines
\newlength{\wordline}\setlength{\wordline}{13.8pt}   % one Word single line
\newlength{\worddouble}\setlength{\worddouble}{27.6pt}
\AtBeginDocument{\budoublespacing}

% Paragraphs: 0.5in first-line indent, no extra space between paragraphs
\setlength{\parindent}{0.5in}
\setlength{\parskip}{0pt}
\RequirePackage{indentfirst}
\raggedbottom
% Word paragraphs are left-aligned (ragged right), not justified
\RequirePackage{ragged2e}
\setlength{\RaggedRightParindent}{0.5in}
\setlength{\RaggedRightRightskip}{0pt plus 1fil}
\hyphenpenalty=10000 % Word does not hyphenate
\AtBeginDocument{\RaggedRight}

% ---------------------------------------------------------------------
%  Template instruction notes (italic grey-box-free text as in Word).
%  Everything wrapped in \templatenote{...} disappears with [nonotes].
% ---------------------------------------------------------------------
\newcommand{\templatenote}[1]{\ifbu@shownotes#1\fi}

% ---------------------------------------------------------------------
%  Page styles
%   prelims : lower-case roman numeral, centred in the footer
%   main    : arabic numeral, centred in the header (every page,
%             including chapter-opening pages)
% ---------------------------------------------------------------------
\fancypagestyle{buprelim}{%
  \fancyhf{}\fancyfoot[C]{\thepage}%
  \renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0pt}}
\fancypagestyle{bumain}{%
  \fancyhf{}\fancyhead[C]{\thepage}%
  \renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0pt}}

\newcommand{\prelims}{%
  \pagenumbering{roman}%
  \fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\thepage}%
    \renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0pt}}%
  \pagestyle{buprelim}}
\newcommand{\mainbody}{%
  \clearpage
  \pagenumbering{arabic}%
  \fancypagestyle{plain}{\fancyhf{}\fancyhead[C]{\thepage}%
    \renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0pt}}%
  \pagestyle{bumain}}
\newcommand{\backmatter}{\clearpage}

% ---------------------------------------------------------------------
%  Headings
%   Heading 1: 12pt bold, centred, double spaced, starts a new page
%   Heading 2: 12pt bold, flush left, double spaced
%   Heading 3: 12pt italic, flush left, 10pt before, double spaced
%   Chapters print as "CHAPTER ONE", "CHAPTER TWO", ...
% ---------------------------------------------------------------------
\setcounter{secnumdepth}{0}      % sections are not numbered
\setcounter{tocdepth}{2}         % TOC shows chapters, sections, subsections

\newcommand{\buchapword}[1]{\ifcase#1\or ONE\or TWO\or THREE\or FOUR\or
  FIVE\or SIX\or SEVEN\or EIGHT\or NINE\or TEN\or ELEVEN\or TWELVE\or
  THIRTEEN\or FOURTEEN\or FIFTEEN\or SIXTEEN\or SEVENTEEN\or EIGHTEEN\or
  NINETEEN\or TWENTY\else\number#1\fi}
% optional chapter title printed on a second line: \chapter{INTRODUCTION}
\newcommand{\bu@ifempty}[3]{\setbox0\hbox{#1}\ifdim\wd0=0pt #2\else #3\fi}
\newcommand{\bu@chaptitle}[1]{\bu@ifempty{#1}{}{\\#1}}

\titleformat{\chapter}[block]
  {\normalfont\normalsize\bfseries\filcenter}
  {CHAPTER \buchapword{\value{chapter}}}{0pt}{\bu@chaptitle}
\titleformat{name=\chapter,numberless}[block]
  {\normalfont\normalsize\bfseries\filcenter}{}{0pt}{}
% titlesec leaves one blank line above the heading; remove it so the
% heading sits at the 1.5in top margin exactly as in Word
\titlespacing*{\chapter}{0pt}{-\baselineskip}{0pt}

\titleformat{\section}[block]{\normalfont\normalsize\bfseries\filright}{}{0pt}{}
\titlespacing*{\section}{0pt}{0pt}{0pt}

\titleformat{\subsection}[block]{\normalfont\normalsize\itshape\filright}{}{0pt}{}
\titlespacing*{\subsection}{0pt}{10pt}{0pt}

% Chapter-level heading for preliminary / back-matter pages.
% Adds the heading to the Table of Contents.
\newcommand{\buheading}[1]{%
  \chapter*{#1}\phantomsection\addcontentsline{toc}{chapter}{#1}}
% Same, but not listed in the TOC
\newcommand{\buheadingnotoc}[1]{\chapter*{#1}}

% ---------------------------------------------------------------------
%  Tables, figures, illustrations and captions
%   Word caption: Times 11pt bold, flush left, single spaced, 10pt after,
%   placed *below* the table / figure; numbering is continuous (1, 2, 3).
% ---------------------------------------------------------------------
\DeclareFloatingEnvironment[
  fileext=loi, listname={LIST OF ILLUSTRATIONS},
  name=Illustration, placement=htbp, within=none]{illustration}
\counterwithout{table}{chapter}
\counterwithout{figure}{chapter}

\captionsetup{font={small,bf,stretch=0.9517}, labelfont=bf, labelsep=period,
  justification=raggedright, singlelinecheck=false,
  position=below, skip=2pt}
\captionsetup[table]{position=below}
\captionsetup[figure]{name=Figure}
\setlength{\belowcaptionskip}{0pt}
% [H] tables/figures sit directly in the text, as in Word
\setlength{\intextsep}{4pt plus 2pt minus 2pt}

% Table cells: single spaced with 2pt padding (Word: 2pt before/after)
\renewcommand{\arraystretch}{1.0}
\AtBeginEnvironment{tabular}{\businglespacing}
\AtBeginEnvironment{tabularx}{\businglespacing}

% ---------------------------------------------------------------------
%  Table of Contents and lists
%   TOC 1: flush left, dot leader, single spaced + 12pt after
%   TOC 2: indented 0.17in ; TOC 3: indented 0.33in
%   List entries: "Table 1. Caption." with 0.33in hanging indent
% ---------------------------------------------------------------------
\renewcommand{\contentsname}{TABLE OF CONTENTS}
\renewcommand{\listtablename}{LIST OF TABLES}
\renewcommand{\listfigurename}{LIST OF FIGURES}
\contentsmargin{2.55em}
\newcommand{\bu@dots}{\titlerule*[0.25em]{.}\contentspage}
\newcommand{\bu@toctitle}[1]{\bu@ifempty{#1}{}{: #1}}

\titlecontents{chapter}[0pt]{}
  {CHAPTER \buchapword{\thecontentslabel}\bu@toctitle}{}{\bu@dots}[\addvspace{12pt}]
\titlecontents{section}[0.17in]{}{}{}{\bu@dots}[\addvspace{12pt}]
\titlecontents{subsection}[0.33in]{}{}{}{\bu@dots}[\addvspace{12pt}]
\titlecontents{table}[0.33in]{}
  {\hspace*{-0.33in}Table \thecontentslabel.\ }{\hspace*{-0.33in}}{\bu@dots}[\addvspace{12pt}]
\titlecontents{figure}[0.33in]{}
  {\hspace*{-0.33in}Figure \thecontentslabel.\ }{\hspace*{-0.33in}}{\bu@dots}[\addvspace{12pt}]
\titlecontents{illustration}[0.33in]{}
  {\hspace*{-0.33in}Illustration \thecontentslabel.\ }{\hspace*{-0.33in}}{\bu@dots}[\addvspace{12pt}]

\renewcommand{\tableofcontents}{%
  \buheading{\contentsname}{\businglespacing\@starttoc{toc}}}
\renewcommand{\listoftables}{%
  \buheading{\listtablename}{\businglespacing\@starttoc{lot}}}
\renewcommand{\listoffigures}{%
  \buheading{\listfigurename}{\businglespacing\@starttoc{lof}}}
\renewcommand{\listofillustrations}{%
  \buheading{LIST OF ILLUSTRATIONS}{\businglespacing\@starttoc{loi}}}

% Abbreviation entry with dot leader:  \abbrev{BU}{Boston University}
\newcommand{\abbrev}[2]{%
  {\businglespacing\noindent#1\titlerule*[0.25em]{.}#2\par}\addvspace{12pt}}

% ---------------------------------------------------------------------
%  Bibliography: single spaced, 12pt between entries, 0.25in hanging
% ---------------------------------------------------------------------
\setlength{\bibsep}{12pt}
\setlength{\bibhang}{0.25in}
\renewcommand{\bibname}{BIBLIOGRAPHY}
\renewcommand{\bibsection}{\buheading{\bibname}}
\renewcommand{\bibfont}{\businglespacing}

% Paragraph style used for the Curriculum Vitae / bibliography notes:
% single spaced, 12pt after, 0.5in hanging indent
\newenvironment{hangingparas}[1][0.5in]
  {\par\businglespacing\setlength{\parindent}{0pt}%
   \setlength{\parskip}{12pt}%
   \everypar{\hangindent=#1\hangafter=1}}
  {\par}

% ---------------------------------------------------------------------
%  Document information
% ---------------------------------------------------------------------
\newcommand{\bu@school}{YOUR FULL SCHOOL/COLLEGE NAME HERE}
\newcommand{\bu@doctype}{Thesis}
\newcommand{\bu@title}{THE TITLE}
\newcommand{\bu@abstracttitle}{\bu@title}
\newcommand{\bu@author}{YOUR NAME}
\newcommand{\bu@authorfull}{\bu@author}
\newcommand{\bu@priordegrees}{}
\newcommand{\bu@degree}{Master of \rule{0.6in}{0.4pt}}
\newcommand{\bu@year}{202X}

\newcommand{\school}[1]{\renewcommand{\bu@school}{#1}}
\newcommand{\doctype}[1]{\renewcommand{\bu@doctype}{#1}}
\renewcommand{\title}[1]{\renewcommand{\bu@title}{#1}}
\newcommand{\abstracttitle}[1]{\renewcommand{\bu@abstracttitle}{#1}}
\renewcommand{\author}[1]{\renewcommand{\bu@author}{#1}}
\newcommand{\authorfull}[1]{\renewcommand{\bu@authorfull}{#1}}
\newcommand{\priordegrees}[1]{\renewcommand{\bu@priordegrees}{#1}}
\newcommand{\degree}[1]{\renewcommand{\bu@degree}{#1}}
\newcommand{\degreeyear}[1]{\renewcommand{\bu@year}{#1}}

% ---------------------------------------------------------------------
%  Title page (page i, not numbered)
% ---------------------------------------------------------------------
\newcommand{\maketitlepage}{%
  \begin{titlepage}
  \businglespacing\centering
  BOSTON UNIVERSITY\par
  \vspace{\wordline}
  \bu@school\par
  \vspace{5\wordline}
  \bu@doctype\par
  \vspace{5\wordline}
  {\bfseries\def\\{\par\vspace{\wordline}}\bu@title\par}
  \vspace{4\wordline}
  by\par
  \vspace{4\wordline}
  {\bfseries\bu@author\par}
  \vspace{\wordline}
  \bu@priordegrees\par
  \vspace{8\wordline}
  Submitted in partial fulfillment of the\par
  \vspace{\wordline}
  requirements for the degree of\par
  \vspace{\wordline}
  \bu@degree\par
  \vspace{\wordline}
  \bu@year\par
  \end{titlepage}%
  \setcounter{page}{2}}

% ---------------------------------------------------------------------
%  Copyright page (page ii, not numbered)
% ---------------------------------------------------------------------
\newcommand{\makecopyrightpage}{%
  \clearpage\thispagestyle{empty}%
  {\businglespacing\setlength{\parindent}{0pt}%
   \null\vfill
   \hspace*{2.75in}\makebox[0.25in][l]{\textbf{\textcopyright}}\bu@year\ by\par
   \hspace*{3in}\bu@authorfull\par
   \hspace*{3in}All rights reserved\par\vspace*{10pt}}%
  \clearpage}

% ---------------------------------------------------------------------
%  Approval page (page iii, not numbered)
%    \begin{approvalpage}
%      \reader{First Reader}{Name, Ph.D.}{Professor of ...}
%    \end{approvalpage}
% ---------------------------------------------------------------------
\newif\ifbu@firstreader
\newenvironment{approvalpage}
  {\clearpage\thispagestyle{empty}%
   \businglespacing\setlength{\parindent}{0pt}%
   \bu@firstreadertrue
   {\centering Approved by\par}%
   \vspace{7\wordline}}
  {\par\clearpage}
% #1 reader label, #2 name, #3 title/affiliation lines (separate with \\)
\newcommand{\reader}[3]{%
  \ifbu@firstreader\bu@firstreaderfalse\else\vspace{4\wordline}\fi
  \makebox[1.25in][l]{#1}\rule[-2pt]{4.625in}{0.6pt}\par
  \begingroup\leftskip=1.25in\relax #2\\#3\par\endgroup}

% ---------------------------------------------------------------------
%  Abstract page: title, author, "ABSTRACT" (not bold), double spaced
% ---------------------------------------------------------------------
\newenvironment{buabstract}
  {\clearpage\thispagestyle{plain}%
   \phantomsection\addcontentsline{toc}{chapter}{ABSTRACT}%
   {\centering\bfseries\bu@abstracttitle\par
    \bu@authorfull\par}%
   {\centering ABSTRACT\par}}
  {\par}

% ---------------------------------------------------------------------
%  hyperref last
% ---------------------------------------------------------------------
\RequirePackage[hidelinks,bookmarksnumbered=false]{hyperref}
\RequirePackage{bookmark}

\endinput
