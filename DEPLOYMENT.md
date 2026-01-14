# EKS 배포 가이드

## 배포 순서

```
사전 작업 → 개발 환경 → 스테이지 환경 → 운영 환경
```

---

## 1. 사전 작업 (완료)

- [x] Dockerfile 작성
- [x] GitHub Actions Workflow 작성
- [x] Helm Chart 작성

---

## 2. 개발 환경 구축

### 2-1. AWS 리소스 생성 (직접 생성)

#### ECR Repository
```
Repository Name: carsharing-vehicle-manual-service
Visibility: Private
```

#### Route 53
```
Record Name: vehicle-manual-dev.socar.me
Record Type: CNAME
Record Value: dualstack.internal-socar-internal-nginx-ingress-alb-xxx.ap-northeast-2.elb.amazonaws.com
(internal - 쏘팸 전용)
```

#### IAM Role
```
Role Name: carsharing-vehicle-manual-service-dev-role

Permission (최소 권한):
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": [
        "arn:aws:secretsmanager:ap-northeast-2:<DEV_ACCOUNT_ID>:secret:carsharing-vehicle-manual-service/dev*"
      ]
    }
  ]
}

Trust Relationship:
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<DEV_ACCOUNT_ID>:oidc-provider/oidc.eks.ap-northeast-2.amazonaws.com/id/<OIDC_ID>"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.ap-northeast-2.amazonaws.com/id/<OIDC_ID>:sub": "system:serviceaccount:<NAMESPACE>:carsharing-vehicle-manual-service-dev"
        }
      }
    }
  ]
}
```

#### Secrets Manager
```
Secret Name: carsharing-vehicle-manual-service/dev
Secret Values (담당자 DM으로 전달):
{
  "OPENAI_API_KEY": "sk-..."
}
```

### 2-2. Helm Chart Repository 설정

1. 팀 Helm Chart 레포에 `helm/carsharing-vehicle-manual-service/` 복사
2. ArgoCD Repositories에 등록 확인

### 2-3. ArgoCD Application 생성

```yaml
project: <PROJECT_NAME>
source:
  repoURL: 'git@github.com:socar-inc/<HELM_CHART_REPO>.git'
  path: carsharing-vehicle-manual-service
  targetRevision: main
  helm:
    valueFiles:
      - values.yaml
      - values-dev.yaml
destination:
  server: 'https://kubernetes.default.svc'
  namespace: <NAMESPACE>
```

### 2-4. GitHub Secrets 등록

Repository Settings > Secrets and variables > Actions:

| Secret Name | 값 |
|-------------|-----|
| `GH_TOKEN` | GitHub Personal Access Token |
| `DEV_ECR_REGISTRY` | `<DEV_ACCOUNT_ID>.dkr.ecr.ap-northeast-2.amazonaws.com` |
| `DEV_AWS_ACCESS_KEY_ID` | AWS Access Key |
| `DEV_AWS_SECRET_ACCESS_KEY` | AWS Secret Key |
| `DEV_HELM_CHART_REPO` | `git@github.com:socar-inc/<HELM_REPO>.git` |
| `DEV_ARGOCD_SERVER` | ArgoCD 서버 URL |
| `DEV_ARGOCD_TOKEN` | ArgoCD API Token |
| `DEV_ARGOCD_APP` | `carsharing-vehicle-manual-service-dev` |

### 2-5. 배포 테스트

```bash
# develop 브랜치에 push → 자동 배포
git checkout develop
git push origin develop

# 확인
kubectl get pods -n <NAMESPACE>
kubectl logs -f deployment/carsharing-vehicle-manual-service -n <NAMESPACE>
```

---

## 3. 스테이지 환경 (Request Ticket)

### 3-1. AWS 리소스 생성 요청

Cloud Infra 팀에 Request Ticket 작성:

**컴포넌트**: `AWS`

```
[요청 리소스 목록]

1. ECR Repository
   - Repository Name: carsharing-vehicle-manual-service
   - Visibility: Private

2. Route 53
   - Record Name: vehicle-manual-stg.socar.me
   - Record Type: CNAME
   - Record Value: internal (쏘팸 전용)

3. IAM Role
   - Role Name: carsharing-vehicle-manual-service-stg-role
   - Permission: (위 개발환경과 동일 구조, Account ID만 변경)
   - Trust Relationship: (위 개발환경과 동일 구조)

4. Secrets Manager
   - Secret Name: carsharing-vehicle-manual-service/stg
   - Secret Values: (담당자 DM으로 전달)
```

### 3-2. ArgoCD Application 생성 요청

**컴포넌트**: `CI/CD`

```
[요청 사항]

1. Helm Chart Repository 등록 (필요시)
   - Repository Name: <HELM_CHART_REPO>
   - Repository URL: git@github.com:socar-inc/<HELM_CHART_REPO>.git

2. ArgoCD Application 생성
   - 테스트된 개발 환경 Application URL: <DEV_ARGOCD_APP_URL>
   - values 파일: values.yaml, values-stg.yaml

project: <PROJECT_NAME>
source:
  repoURL: 'git@github.com:socar-inc/<HELM_CHART_REPO>.git'
  path: carsharing-vehicle-manual-service
  targetRevision: main
  helm:
    valueFiles:
      - values.yaml
      - values-stg.yaml
destination:
  server: 'https://kubernetes.default.svc'
  namespace: <NAMESPACE>
```

---

## 4. 운영 환경 (Request Ticket)

### 4-1. AWS 리소스 생성 요청

(스테이지와 동일 구조, 환경 변수만 prod로 변경)

### 4-2. ArgoCD Application 생성 요청

(스테이지와 동일 구조, values-prod.yaml 사용)

---

## 배포 방법

### 개발 환경
```bash
# develop 브랜치에 push → 자동 배포
git push origin develop
```

### 스테이지/운영 환경
```bash
# 1. Git Tag 생성 (v prefix 필수)
git tag v1.0.0
git push origin v1.0.0

# 2. GitHub Actions > workflow_dispatch 실행
# - Tag 선택: v1.0.0
# - Jira Ticket 입력: SD-XXX
```

---

## 트러블슈팅

### Pod가 시작되지 않을 때
```bash
# Pod 상태 확인
kubectl describe pod <POD_NAME> -n <NAMESPACE>

# 로그 확인
kubectl logs <POD_NAME> -n <NAMESPACE>
```

### Secrets 오류
```bash
# ExternalSecret 상태 확인
kubectl get externalsecret -n <NAMESPACE>
kubectl describe externalsecret <NAME> -n <NAMESPACE>
```

### Ingress 연결 오류
```bash
# Ingress 상태 확인
kubectl get ingress -n <NAMESPACE>
kubectl describe ingress <NAME> -n <NAMESPACE>
```
