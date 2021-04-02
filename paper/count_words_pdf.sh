#!/bin/bash

pdftotext "$1" - | sed -n "/Abstract/,/References/p" | wc -w
