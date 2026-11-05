# XeLaTeX。在项目根目录执行：latexmk eigenfaces_algorithm.tex
$pdf_mode = 5;
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error %O %S';
$clean_ext .= ' synctex.gz synctex.gz(busy) xdv run.xml';
