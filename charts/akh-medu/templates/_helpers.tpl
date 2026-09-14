{{/*
Expand the name of the chart.
*/}}
{{- define "akh-medu.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Mandatory shared ingress-auth Secret name. */}}
{{- define "akh-medu.ingressAuth.secretName" -}}
{{- if .Values.ingressAuth.existingSecret -}}
{{- .Values.ingressAuth.existingSecret -}}
{{- else -}}
{{- $_ := required "ingressAuth.sharedSecret is required when ingressAuth.existingSecret is empty" .Values.ingressAuth.sharedSecret -}}
{{- printf "%s-ingress-auth" (include "akh-medu.fullname" .) -}}
{{- end -}}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "akh-medu.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart label boilerplate.
*/}}
{{- define "akh-medu.labels" -}}
helm.sh/chart: {{ include "akh-medu.chart" . }}
{{ include "akh-medu.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "akh-medu.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "akh-medu.selectorLabels" -}}
app.kubernetes.io/name: {{ include "akh-medu.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
akhomed component labels.
*/}}
{{- define "akh-medu.akhomed.labels" -}}
{{ include "akh-medu.labels" . }}
app.kubernetes.io/component: akhomed
{{- end }}

{{- define "akh-medu.akhomed.selectorLabels" -}}
{{ include "akh-medu.selectorLabels" . }}
app.kubernetes.io/component: akhomed
{{- end }}

{{/*
seshd component labels.
*/}}
{{- define "akh-medu.seshd.labels" -}}
{{ include "akh-medu.labels" . }}
app.kubernetes.io/component: seshd
{{- end }}

{{- define "akh-medu.seshd.selectorLabels" -}}
{{ include "akh-medu.selectorLabels" . }}
app.kubernetes.io/component: seshd
{{- end }}

{{/*
sokard component labels.
*/}}
{{- define "akh-medu.sokard.labels" -}}
{{ include "akh-medu.labels" . }}
app.kubernetes.io/component: sokard
{{- end }}

{{- define "akh-medu.sokard.selectorLabels" -}}
{{ include "akh-medu.selectorLabels" . }}
app.kubernetes.io/component: sokard
{{- end }}

{{/*
akhomed image reference.
*/}}
{{- define "akh-medu.akhomed.image" -}}
{{- $registry := .Values.global.imageRegistry -}}
{{- $repo := .Values.akhomed.image.repository -}}
{{- $tag := .Values.akhomed.image.tag | default .Chart.AppVersion -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end }}

{{/*
seshd image reference.
*/}}
{{- define "akh-medu.seshd.image" -}}
{{- $registry := .Values.global.imageRegistry -}}
{{- $repo := .Values.seshd.image.repository -}}
{{- $tag := .Values.seshd.image.tag | default .Chart.AppVersion -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end }}

{{/*
sokard image reference.
*/}}
{{- define "akh-medu.sokard.image" -}}
{{- $registry := .Values.global.imageRegistry -}}
{{- $repo := .Values.sokard.image.repository -}}
{{- $tag := .Values.sokard.image.tag | default .Chart.AppVersion -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end }}

{{/*
Architecture affinity — pins pods to nodes whose kubernetes.io/arch matches
one of the architectures we have actually published images for. Merges with
any user-supplied affinity via `toYaml` in the caller.
*/}}
{{- define "akh-medu.archAffinity" -}}
nodeAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    nodeSelectorTerms:
      - matchExpressions:
          - key: kubernetes.io/arch
            operator: In
            values:
              {{- range .Values.supportedArchitectures }}
              - {{ . | quote }}
              {{- end }}
          - key: kubernetes.io/os
            operator: In
            values:
              - "linux"
{{- end }}

{{/*
PostgreSQL host.
Subchart: <release>-postgresql. External: externalPostgresql.host.
*/}}
{{- define "akh-medu.postgresql.host" -}}
{{- if .Values.postgresql.enabled -}}
{{- printf "%s-postgresql" (include "akh-medu.fullname" .) -}}
{{- else -}}
{{- .Values.externalPostgresql.host -}}
{{- end -}}
{{- end }}

{{/*
PostgreSQL port.
*/}}
{{- define "akh-medu.postgresql.port" -}}
{{- if .Values.postgresql.enabled -}}
5432
{{- else -}}
{{- .Values.externalPostgresql.port | default 5432 -}}
{{- end -}}
{{- end }}

{{/*
PostgreSQL database name.
*/}}
{{- define "akh-medu.postgresql.database" -}}
{{- if .Values.postgresql.enabled -}}
{{- .Values.postgresql.auth.database -}}
{{- else -}}
{{- .Values.externalPostgresql.database -}}
{{- end -}}
{{- end }}

{{/*
PostgreSQL username.
*/}}
{{- define "akh-medu.postgresql.username" -}}
{{- if .Values.postgresql.enabled -}}
{{- .Values.postgresql.auth.username -}}
{{- else -}}
{{- .Values.externalPostgresql.username -}}
{{- end -}}
{{- end }}

{{/*
PostgreSQL password (plaintext, for constructing the secret).
*/}}
{{- define "akh-medu.postgresql.password" -}}
{{- if .Values.postgresql.enabled -}}
{{- .Values.postgresql.auth.password -}}
{{- else -}}
{{- .Values.externalPostgresql.password -}}
{{- end -}}
{{- end }}

{{/*
Name of the secret containing DATABASE_URL.
When using an external existing secret, that secret must contain a DATABASE_URL key.
*/}}
{{- define "akh-medu.db.secretName" -}}
{{- if and (not .Values.postgresql.enabled) .Values.externalPostgresql.existingSecret -}}
{{- .Values.externalPostgresql.existingSecret -}}
{{- else -}}
{{- printf "%s-seshd-db" (include "akh-medu.fullname" .) -}}
{{- end -}}
{{- end }}

{{/*
Name of the secret containing SOKAR_DATABASE_URL (or whatever
sokard.database.secretKey names). Distinct from seshd's db secret: sokar
gets its own database/user (default "sokar"/"sokar") on the same PostgreSQL
host — see sokard.database in values.yaml.
*/}}
{{- define "akh-medu.sokard.db.secretName" -}}
{{- if .Values.sokard.database.existingSecret -}}
{{- .Values.sokard.database.existingSecret -}}
{{- else -}}
{{- printf "%s-sokard-db" (include "akh-medu.fullname" .) -}}
{{- end -}}
{{- end }}
