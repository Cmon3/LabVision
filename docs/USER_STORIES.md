# User stories — LabVision Copilot

| # | Como… | Quiero… | Se cumple si… | Test |
|---|-------|---------|---------------|------|
| 1 | técnico | contar las colonias de una placa a partir de una foto | sale un conteo con máscaras redondas sobre cada colonia | `tests/test_server.py::test_analyze_ok` |
| 2 | técnico con guantes | pedirlo por voz | lo que digo aparece en el cuadro de petición | `tests/test_voice.py::test_transcribe_mocked` |
| 3 | técnico | filtrar por tamaño (“larger than 2 mm”) | el conteo cambia y los tamaños están en mm | `tests/test_measure.py::test_measure_mm_scale_and_filter_2mm` |
| 4 | técnico | saber si la foto sirve | una foto borrosa da “Re-photo” y no un número | `tests/test_qc.py::test_blur_flag` |
| 5 | técnico | que no me dé un número falso en placas saturadas | más de 300 colonias, o una placa densa con pocas detecciones, va a revisión | `tests/test_qc.py::test_crowded_flags` |
| 6 | técnico | corregir una colonia que se saltó | al dibujar una caja se añade, sube el conteo y queda como “human” | `tests/test_server.py::test_correct_add_remove` |
| 7 | técnico | no inventar colonias | una caja sobre agar vacío da un aviso y no añade nada | `tests/test_server.py::test_correct_no_fabricated_mask` |
| 8 | revisor | revisar solo lo dudoso | la cola pone arriba las marcadas, con el motivo en palabras claras | `tests/test_queue.py::test_queue_needs_review_and_sort` |
| 9 | revisor | aprobar y dejar constancia | aprobar exige nombre y queda con fecha en el registro | `tests/test_queue.py::test_queue_needs_review_and_sort` |
| 10 | laboratorio | llevar los datos al LIMS | el CSV tiene una fila por placa con conteo, estado y revisor | `tests/test_runlog.py::test_save_creates_json_and_csv` |
| 11 | laboratorio con cámara fija | no subir fotos a mano | una foto en inbox/ aparece sola en la cola, con el ID leído del QR | `tests/test_inbox.py::test_process_success_and_move` |
| 12 | auditor | ver cómo se obtuvo cada número | la línea de tiempo muestra cada paso con model/rule/human, y el informe es rechazado si no cuadra | `tests/test_verification.py::test_verify_wrong_count_refused` |
| 13 | integrador | conectar otro equipo | /docs documenta la API y /api/analyze responde | `tests/test_server.py::test_analyze_ok` |

*Manual where no automated test exists: none — all stories above have a related automated check. Use `Try it in 5 minutes` in README for end-to-end manual verification.*
