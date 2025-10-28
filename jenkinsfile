pipeline {
    agent any
    options {
        timestamps()
        ansiColor('xterm')
    }

    stages {
        stage('Checkout') {
            steps {
                echo '📦 Obteniendo código del workspace montado...'
                // No hace falta checkout si ya tienes los archivos montados en el contenedor
                sh 'ls -l'
            }
        }

        stage('Preparar entorno Python') {
            steps {
                echo '🐍 Creando entorno virtual y preparando dependencias...'
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt || true
                '''
            }
        }

        stage('Ejecutar validaciones PyOps') {
            steps {
                echo '🔍 Ejecutando validaciones de datasets...'
                sh '''
                    . venv/bin/activate
                    python3 pyops/validate.py > reports.log
                '''
            }
        }

        stage('Mostrar resultados') {
            steps {
                echo '📋 Resultados de validación:'
                sh 'cat reports.log || true'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'reports.log', fingerprint: true
        }
        success {
            echo '✅ Validaciones completadas correctamente.'
        }
        failure {
            echo '❌ Fallaron las validaciones. Revisa el log reports.log.'
        }
    }
}
