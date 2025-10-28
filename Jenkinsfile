pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        REPORT_FILE = "reports.log"
    }

    stages {

        stage('Checkout') {
            steps {
                echo "📦 Obteniendo código del workspace montado..."
                sh 'ls -l'
            }
        }

        stage('Preparar entorno Python') {
            steps {
                echo "🐍 Creando entorno virtual y preparando dependencias..."
                sh '''
                    apt-get update
                    apt-get install -y python3-venv
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Ejecutar validaciones PyOps') {
            steps {
                echo "🔍 Ejecutando validaciones de datasets..."
                sh '''
                    . venv/bin/activate
                    python3 pyops/validate.py | tee ${REPORT_FILE}
                '''
            }
        }

        // 🆕 Nueva etapa añadida
        stage('Validación de seguridad global') {
            steps {
                echo "🛡️ Buscando posibles llaves o credenciales filtradas..."
                sh '''
                    . venv/bin/activate
                    python3 pyops/validate_leaks.py | tee -a ${REPORT_FILE}
                '''
            }
        }

        stage('Mostrar resultados') {
            steps {
                echo "📋 Resultados de validación:"
                sh 'cat ${REPORT_FILE}'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**/reports.log', fingerprint: true
        }
        success {
            echo "✅ Validaciones completadas correctamente."
        }
        failure {
            echo "❌ Fallaron las validaciones. Revisa el log reports.log."
        }
    }
}

