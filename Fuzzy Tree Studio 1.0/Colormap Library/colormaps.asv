%% Con estas dos abrimos el editor de colormap para 256 colores.
%  No hay que cerrar la figura, por supuesto.

% colormap(jet(256))
% colormapeditor

%% Grabamos el colormap generado.
a = get(gcf, 'Colormap')
% save('semaforoGus.csv', 'a', '-ascii', '-tabs')
cellArray = num2cell(a)
separator = ';';
excelYear = 1900;
decimal = ',';
cell2csv('semaforoGustavo.csv', cellArray, separator, excelYear, decimal)
