"""
Ejemplo práctico: Extracción de Informes Policiales e Incidentes Forenses.
Demuestra la conexión entre Criminología / Investigación y AI Engineering.
"""

from typing import Literal
from pydantic import BaseModel, Field
from guardrails import SelfHealingEngine, GeminiClient


class Suspect(BaseModel):
    alias_or_name: str = Field(description="Nombre o alias conocido del sospechoso")
    physical_description: str = Field(description="Rasgos físicos descritos")
    status: Literal["Identificado", "Detenido", "En fuga", "Desconocido"]


class Evidence(BaseModel):
    item: str = Field(description="Descripción del elemento o evidencia física")
    location_found: str = Field(description="Lugar específico donde fue hallada")


class IncidentReport(BaseModel):
    incident_type: str = Field(description="Tipo de delito o incidente investigado")
    date_approx: str = Field(description="Fecha o momento temporal aproximado")
    location: str = Field(description="Ubicación de los hechos")
    risk_level: Literal["Bajo", "Medio", "Alto", "Crítico"]
    suspects: list[Suspect] = Field(description="Lista de sospechosos identificados")
    evidences: list[Evidence] = Field(description="Lista de evidencias recolectadas")
    summary: str = Field(description="Síntesis narrativa del hecho")


def main():
    sample_police_narrative = """
    REPORTE DE PATRULLAJE - FECHA: 18 de Agosto 2026, 23:45 hrs.
    Nos trasladamos al local comercial 'Supermercado Central' ubicado en Av. Libertador #104.
    El encargado reportó que dos individuos ingresaron armados. Uno de ellos, conocido en la zona
    con el alias de 'El Flaco' (delgado, aprox 1.80m, campera oscura), escapó del lugar con dinero en efectivo.
    El segundo sujeto fue reducido por el personal de seguridad en el interior.
    En el suelo del estacionamiento se recuperó un arma de fuego calibre 9mm y una mochila negra
    que contenía herramientas de corte. Clasificado con prioridad máxima de investigación.
    """

    print("Inicializando Self-Healing Guardrails Engine...")
    try:
        client = GeminiClient(model="gemini-2.5-flash")
        engine = SelfHealingEngine(llm_client=client, max_retries=2, prefer_native=True)

        result = engine.extract(sample_police_narrative, IncidentReport)

        if result.success and result.data:
            print("\n✅ Extracción exitosa!")
            print(f"Delito: {result.data.incident_type}")
            print(f"Nivel de Riesgo: {result.data.risk_level}")
            print(f"Sospechosos: {len(result.data.suspects)}")
            for s in result.data.suspects:
                print(f"  - {s.alias_or_name} [{s.status}]: {s.physical_description}")
            print(f"Evidencias: {len(result.data.evidences)}")
            for e in result.data.evidences:
                print(f"  - {e.item} (Ubicación: {e.location_found})")
            print(f"\nIntentos requeridos: {result.attempts}")
            print(f"Tokens totales: {result.total_tokens.total_tokens}")
        else:
            print(f"\n❌ Error en extracción: {result.error}")
    except Exception as e:
        print(f"Error de ejecución: {e}")


if __name__ == "__main__":
    main()
