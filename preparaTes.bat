@echo off
if exist "resumen" rmdir /s /q "resumen"
if exist "faq" rmdir /s /q "faq"
if exist "json" rmdir /s /q "json"
if exist "actas_procesadas" rmdir /s /q "actas_procesadas"
if exist "log\pipeline.log" del /q "log\pipeline.log"