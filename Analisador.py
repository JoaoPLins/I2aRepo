import sys
import zipfile
import pandas as pd
from typing import Callable
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, QLabel, 
                            QLineEdit, QPushButton, QTextEdit, QFileDialog,
                            QHBoxLayout, QProgressBar, QApplication, QFontDialog, 
                            QAction)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette, QKeySequence
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
#from langchain.agents import create_tool_calling_agent
#from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
import os
from pydantic import BaseModel
from dotenv import load_dotenv
import warnings

# Filtrar avisos de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

class DarkPalette:
    @staticmethod
    def get_dark_palette():
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.HighlightedText, Qt.black)
        return palette

class DataLoader:
    def __init__(self):
        self.cabecalho_df = None
        self.itens_df = None
        self.merged_df = None
        
    def load_data(self, zip_path: str, progress_callback: Callable = None):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if progress_callback:
                progress_callback(10)
                
            with zip_ref.open('202401_NFs_Cabecalho.csv') as f:
                self.cabecalho_df = pd.read_csv(f, encoding='utf-8', sep=',')
                
            if progress_callback:
                progress_callback(40)
                
            with zip_ref.open('202401_NFs_Itens.csv') as f:
                self.itens_df = pd.read_csv(f, encoding='utf-8', sep=',')
                
            if progress_callback:
                progress_callback(70)
                
            self.merged_df = pd.merge(
                self.cabecalho_df,
                self.itens_df,
                on='CHAVE DE ACESSO',
                how='left'
            )
            
            if progress_callback:
                progress_callback(100)
    
    def get_summary(self) -> str:
        if self.merged_df is None:
            return "Nenhum dado carregado"
            
        columns_description = """
        ## Estrutura dos Dados de Notas Fiscais:

        ### Arquivo de Cabeçalho (202401_NFs_Cabecalho.csv):
        - CHAVE DE ACESSO: Identificador único da NF-e
        - MODELO: Modelo do documento fiscal (ex: 55 - NF-e)
        - SÉRIE: Série da nota fiscal
        - NÚMERO: Número da nota fiscal
        - NATUREZA DA OPERAÇÃO: Descrição da operação (Venda, Devolução, etc.)
        - DATA EMISSÃO: Data e hora de emissão da nota
        - EVENTO MAIS RECENTE: Último evento ocorrido com a NF-e
        - DATA/HORA EVENTO MAIS RECENTE: Quando ocorreu o último evento
        - CPF/CNPJ Emitente: Documento do emissor
        - RAZÃO SOCIAL EMITENTE: Nome da empresa emissora
        - INSCRIÇÃO ESTADUAL EMITENTE: IE do emissor
        - UF EMITENTE: Estado do emissor
        - MUNICÍPIO EMITENTE: Cidade do emissor
        - CNPJ DESTINATÁRIO: Documento do destinatário
        - NOME DESTINATÁRIO: Razão social do destinatário
        - UF DESTINATÁRIO: Estado do destinatário
        - INDICADOR IE DESTINATÁRIO: Se destinatário é contribuinte
        - DESTINO DA OPERAÇÃO: 1-Interna, 2-Interestadual
        - CONSUMIDOR FINAL: 0-Não, 1-Sim
        - PRESENÇA DO COMPRADOR: 0-Não se aplica, 1-Presencial, etc.
        - VALOR NOTA FISCAL: Valor total da nota

        ### Arquivo de Itens (202401_NFs_Itens.csv):
        - NÚMERO PRODUTO: Número sequencial do item
        - DESCRIÇÃO DO PRODUTO/SERVIÇO: Nome detalhado do produto
        - CÓDIGO NCM/SH: Código da Nomenclatura Comum do Mercosul
        - NCM/SH (TIPO DE PRODUTO): Descrição do tipo de produto
        - CFOP: Código Fiscal de Operações e Prestações
        - QUANTIDADE: Quantidade do item
        - UNIDADE: Unidade de medida (UN, KG, L, etc.)
        - VALOR UNITÁRIO: Preço unitário do item
        - VALOR TOTAL: Valor total do item (quantidade × unitário)
        """

        stats_summary = f"""
        ## Estatísticas dos Dados:
        - Total de notas: {len(self.cabecalho_df)}
        - Total de itens: {len(self.itens_df)}
        - Valor total: R$ {self.cabecalho_df['VALOR NOTA FISCAL'].sum():,.2f}
        - Média por nota: R$ {self.cabecalho_df['VALOR NOTA FISCAL'].mean():,.2f}
        - Maior nota: R$ {self.cabecalho_df['VALOR NOTA FISCAL'].max():,.2f}
        - Menor nota: R$ {self.cabecalho_df['VALOR NOTA FISCAL'].min():,.2f}
        - Principais destinatários: {self.cabecalho_df['NOME DESTINATÁRIO'].value_counts().head(3).to_dict()}
        - Principais produtos: {self.itens_df['DESCRIÇÃO DO PRODUTO/SERVIÇO'].value_counts().head(5).to_dict()}
        - Tipos de operação mais comuns: {self.cabecalho_df['NATUREZA DA OPERAÇÃO'].value_counts().head(5).to_dict()}
        - Distribuição por UF emitente: {self.cabecalho_df['UF EMITENTE'].value_counts().to_dict()}
        - Distribuição por UF destinatário: {self.cabecalho_df['UF DESTINATÁRIO'].value_counts().to_dict()}
        """

        return columns_description + stats_summary

class ToolResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    analysis: str

parser = PydanticOutputParser(pydantic_object=ToolResponse)

class GeminiIntegration:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY não encontrada no arquivo .env")
            
        self.llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.3)
        
        self.prompt_template = PromptTemplate(
            input_variables=["question", "data_summary"],
            template="""
            Você é um analista especializado em dados fiscais e notas fiscais eletrônicas (NF-e).
            
            Contexto sobre os dados:
            {data_summary}
            
            Instruções para análise:
            1. Responda com precisão baseando-se exclusivamente nos dados fornecidos
            2. Para cálculos, mostre o raciocínio passo a passo
            3. Ao mencionar valores monetários, formate como R$ 1.234,56
            4. Para operações interestaduais, considere o CFOP e UF de origem/destino
            5. Para produtos, utilize a descrição e NCM quando relevante
            6. wrap the output and provide no other text\n {format_instructions}
            
            Pergunta: {question}
            
            Responda de forma estruturada:
            1. Análise solicitada
            2. Metodologia utilizada
            3. Resultados encontrados
            4. Observações relevantes
            """
        ).partial(format_instructions=parser.get_format_instructions())
        
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        
    def ask_question(self, question: str, data_summary: str) -> str:
        # Método atualizado para usar invoke() em vez de run()
        response = self.chain.invoke({"question": question, "data_summary": data_summary})
        return response['text']

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analisador Avançado de NF-e v2.0")
        self.setGeometry(100, 100, 900, 700)
        
        # Configuração inicial
        self.dark_mode = True
        self.default_font = QFont("Segoe UI", 10)
        
        self.data_loader = DataLoader()
        try:
            self.gemini = GeminiIntegration()
            self.gemini_available = True
        except Exception as e:
            self.gemini_available = False
            print(f"Erro ao inicializar Gemini: {str(e)}")
        
        self.init_ui()
        self.apply_dark_theme()
        self.connect_signals()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Menu superior
        menubar = self.menuBar()
        
        # Menu Configurações
        config_menu = menubar.addMenu('Configurações')
        
        font_action = QAction('Alterar Fonte', self)
        font_action.triggered.connect(self.change_font)
        config_menu.addAction(font_action)
        
        theme_action = QAction('Alternar Tema (Dark/Light)', self)
        theme_action.triggered.connect(self.toggle_theme)
        config_menu.addAction(theme_action)
        
        # Área principal
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Arquivo ZIP não selecionado")
        self.browse_button = QPushButton("Selecionar ZIP")
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.browse_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ex: Quais os 5 produtos mais vendidos por valor total?")
        
        self.ask_button = QPushButton("Analisar")
        self.ask_button.setEnabled(False)
        self.ask_button.setDefault(True)
        
        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setFont(QFont("Consolas", 10))
        
        self.status_label = QLabel()
        if not self.gemini_available:
            self.status_label.setText("Integração com Gemini não disponível - Verifique sua chave API no arquivo .env")
        
        layout.addLayout(file_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Pergunta:"))
        layout.addWidget(self.question_input)
        layout.addWidget(self.ask_button)
        layout.addWidget(QLabel("Resposta da Análise:"))
        layout.addWidget(self.response_area)
        layout.addWidget(self.status_label)
        
        self.apply_font_to_all()
        
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
        
    def apply_dark_theme(self):
        self.setPalette(DarkPalette.get_dark_palette())
        self.response_area.setStyleSheet("""
            QTextEdit {
                background-color: #252525;
                color: #ffffff;
                border: 1px solid #444;
            }
        """)
        
    def apply_light_theme(self):
        self.setPalette(QApplication.style().standardPalette())
        self.response_area.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #ccc;
            }
        """)
        
    def change_font(self):
        font, ok = QFontDialog.getFont(self.default_font, self)
        if ok:
            self.default_font = font
            self.apply_font_to_all()
            
    def apply_font_to_all(self):
        widgets = [
            self.file_label, self.browse_button, self.question_input,
            self.ask_button, self.status_label, self.progress_bar
        ]
        
        for widget in widgets:
            widget.setFont(self.default_font)
        
        central = self.centralWidget()
        for i in range(central.layout().count()):
            widget = central.layout().itemAt(i).widget()
            if isinstance(widget, QLabel):
                widget.setFont(self.default_font)
        
    def connect_signals(self):
        self.browse_button.clicked.connect(self.load_zip_file)
        self.ask_button.clicked.connect(self.ask_question)
        self.question_input.returnPressed.connect(self.ask_button.click)
        
    def load_zip_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecione o arquivo ZIP com as NF-e", "", "Arquivos ZIP (*.zip)")
        
        if file_path:
            self.file_label.setText(f"Arquivo: {file_path}")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            try:
                self.data_loader.load_data(file_path, self.update_progress)
                self.progress_bar.setValue(100)
                self.ask_button.setEnabled(self.gemini_available)
                self.response_area.clear()
                self.response_area.append("Dados carregados com sucesso!\n")
                self.response_area.append("Resumo estatístico disponível para análise.")
            except Exception as e:
                self.response_area.append(f"Erro ao carregar arquivo: {str(e)}")
            finally:
                self.progress_bar.setVisible(False)
            
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def ask_question(self):
        question = self.question_input.text().strip()
        if not question:
            return
            
        self.response_area.clear()
        self.response_area.append(f"[Pergunta] {question}")
        self.response_area.append("Processando análise...")
        QApplication.processEvents()
        
        try:
            summary = self.data_loader.get_summary()
            answer = self.gemini.ask_question(question, summary)
            self.response_area.append(f"\n[Resposta]\n{answer}\n")
            self.response_area.append("-"*80)
        except Exception as e:
            self.response_area.append(f"\n[Erro] Não foi possível processar a pergunta: {str(e)}\n")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()