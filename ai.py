import chromadb
import json
import boto3
import pickle
import gzip
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import openai

class InstagramVectorDBManager:
    def __init__(self, s3_bucket: str, openai_api_key: str):
        self.s3_client = boto3.client('s3')
        self.s3_bucket = s3_bucket
        self.embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        openai.api_key = openai_api_key
        
        # 로컬 임시 디렉토리
        self.local_temp_dir = "/tmp/vector_dbs"
        os.makedirs(self.local_temp_dir, exist_ok=True)
    
    def process_instagram_data(self, json_data: Dict[str, Any]) -> str:
        """
        Instagram JSON 데이터를 처리하여 벡터DB 업데이트 및 요약 생성
        """
        account_id = json_data["accountId"]
        content_id = json_data["contentId"]
        content_url = json_data["contentUrl"]
        comments = json_data["comments"]
        crawled_at = json_data["crawledAt"]
        
        print(f"Processing account {account_id}, content {content_id}")
        
        # 1. 벡터 DB 로드 또는 생성
        collection = self.get_or_create_vectordb(account_id)
        
        # 2. 새 댓글들을 벡터 DB에 추가
        self.add_comments_to_vectordb(collection, comments, content_id, content_url, crawled_at)
        
        # 3. 벡터 DB를 S3에 저장
        self.save_vectordb_to_s3(account_id, collection)
        
        # 4. 계정 요약 생성
        summary = self.generate_account_summary(collection, account_id)
        
        return summary
    
    def get_or_create_vectordb(self, account_id: int):
        """
        계정의 벡터 DB를 로드하거나 새로 생성
        """
        s3_key = f"vector_dbs/account_{account_id}.gz"
        local_path = f"{self.local_temp_dir}/account_{account_id}"
        
        try:
            # S3에서 기존 벡터 DB 다운로드
            print(f"Loading existing vector DB for account {account_id}")
            self.s3_client.download_file(self.s3_bucket, s3_key, f"{local_path}.gz")
            
            # 압축 해제 후 로드
            with gzip.open(f"{local_path}.gz", 'rb') as f:
                db_data = pickle.load(f)
            
            # ChromaDB 클라이언트 생성
            client = chromadb.PersistentClient(path=local_path)
            collection = client.get_or_create_collection(
                name=f"account_{account_id}",
                metadata={"account_id": account_id}
            )
            
            # 기존 데이터 복원
            if db_data:
                collection.add(
                    ids=db_data['ids'],
                    embeddings=db_data['embeddings'],
                    documents=db_data['documents'],
                    metadatas=db_data['metadatas']
                )
            
            print(f"Loaded existing vector DB with {collection.count()} items")
            
        except Exception as e:
            # 벡터 DB가 없으면 새로 생성
            print(f"Creating new vector DB for account {account_id}: {e}")
            client = chromadb.PersistentClient(path=local_path)
            collection = client.get_or_create_collection(
                name=f"account_{account_id}",
                metadata={"account_id": account_id}
            )
        
        return collection
    
    def add_comments_to_vectordb(self, collection, comments: List[Dict], 
                                content_id: int, content_url: str, crawled_at: str):
        """
        새 댓글들을 벡터 DB에 추가
        """
        if not comments:
            print("No comments to add")
            return
        
        # 중복 확인을 위해 기존 ID들 조회
        try:
            existing_data = collection.get()
            existing_ids = set(existing_data['ids']) if existing_data['ids'] else set()
        except:
            existing_ids = set()
        
        new_comments = []
        new_embeddings = []
        new_metadatas = []
        new_ids = []
        
        for comment in comments:
            comment_id = f"content_{content_id}_comment_{comment['externalCommentId']}"
            
            # 중복 댓글 스킵
            if comment_id in existing_ids:
                print(f"Skipping duplicate comment: {comment_id}")
                continue
            
            # 댓글 텍스트 전처리
            comment_text = comment['text'].strip()
            if len(comment_text) < 5:  # 너무 짧은 댓글 스킵
                continue
            
            new_comments.append(comment_text)
            new_metadatas.append({
                "content_id": content_id,
                "content_url": content_url,
                "author": comment['accountNickname'],
                "external_comment_id": comment['externalCommentId'],
                "likes_count": comment['likesCount'],
                "published_at": comment['publishedAt'],
                "crawled_at": crawled_at,
                "added_to_db_at": datetime.now().isoformat()
            })
            new_ids.append(comment_id)
        
        if not new_comments:
            print("No new comments to add")
            return
        
        # 임베딩 생성
        print(f"Generating embeddings for {len(new_comments)} new comments")
        embeddings = self.embedding_model.encode(new_comments)
        new_embeddings = [emb.tolist() for emb in embeddings]
        
        # 벡터 DB에 추가
        collection.add(
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_comments,
            metadatas=new_metadatas
        )
        
        print(f"Added {len(new_comments)} new comments to vector DB")
    
    def save_vectordb_to_s3(self, account_id: int, collection):
        """
        벡터 DB를 S3에 압축 저장
        """
        try:
            # 컬렉션 데이터 추출
            data = collection.get()
            
            db_data = {
                'ids': data['ids'],
                'embeddings': data['embeddings'],
                'documents': data['documents'],
                'metadatas': data['metadatas'],
                'saved_at': datetime.now().isoformat(),
                'total_count': len(data['ids']) if data['ids'] else 0
            }
            
            # 압축해서 로컬에 저장
            local_gz_path = f"{self.local_temp_dir}/account_{account_id}.gz"
            with gzip.open(local_gz_path, 'wb') as f:
                pickle.dump(db_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # S3에 업로드
            s3_key = f"vector_dbs/account_{account_id}.gz"
            self.s3_client.upload_file(local_gz_path, self.s3_bucket, s3_key)
            
            print(f"Saved vector DB to S3: {s3_key}")
            
            # 로컬 임시 파일 정리
            os.remove(local_gz_path)
            
        except Exception as e:
            print(f"Error saving vector DB to S3: {e}")
    
    def generate_account_summary(self, collection, account_id: int) -> str:
        """
        댓글 데이터 특성에 맞는 다층적 분석으로 계정 요약 생성
        """
        try:
            # 전체 댓글 수 확인
            total_count = collection.count()
            
            if total_count < 10:
                return f"계정 {account_id}: 분석하기에 댓글이 부족합니다 ({total_count}개)"
            
            # 전체 댓글 데이터 가져오기
            all_data = collection.get()
            comments = all_data['documents']
            metadatas = all_data['metadatas']
            
            # 방법 1: 클러스터링 기반 대표 댓글 선택
            representative_comments = self.get_representative_comments(comments)
            
            # 방법 2: 감정/특성 기반 분류
            categorized_comments = self.categorize_comments(comments, metadatas)
            
            # 방법 3: 시간 기반 트렌드 분석
            temporal_analysis = self.analyze_temporal_trends(comments, metadatas)
            
            # 최종 요약용 댓글 선별
            selected_comments = (
                representative_comments[:15] +  # 대표 댓글
                categorized_comments['high_engagement'][:10] +  # 인기 댓글
                categorized_comments['recent'][:10] +  # 최신 댓글
                categorized_comments['questions'][:5]  # 질문/문의
            )
            
            # 중복 제거
            unique_comments = list(set(selected_comments))[:40]
            
            # 통계 정보와 함께 LLM 요약 생성
            summary_prompt = f"""
Instagram 계정 {account_id}의 댓글 분석 결과:

📊 통계 요약:
- 총 댓글 수: {total_count}개
- 평균 좋아요: {categorized_comments['avg_likes']:.1f}개
- 주요 시간대: {temporal_analysis['peak_hours']}
- 활동 기간: {temporal_analysis['date_range']}

💬 대표적인 댓글들:
{chr(10).join(unique_comments[:25])}

📈 참여도 높은 댓글:
{chr(10).join(categorized_comments['high_engagement'][:10])}

❓ 질문/문의사항:
{chr(10).join(categorized_comments['questions'][:5])}

위 분석을 바탕으로 이 Instagram 계정의 특성을 요약해주세요:
1. 주요 콘텐츠 특성과 팔로워 관심사
2. 팔로워 참여 패턴과 반응
3. 주요 피드백 및 개선점

4-6문장으로 간결하게 요약해주세요.
"""

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content
            
            # 요약에 메타 정보 추가
            final_summary = f"""
=== Instagram 계정 {account_id} 분석 요약 ===
총 댓글 수: {total_count}개
분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{summary}
"""
            
            print(f"Generated summary for account {account_id}")
            return final_summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"계정 {account_id} 요약 생성 중 오류 발생: {str(e)}"
    
    def get_representative_comments(self, comments: List[str]) -> List[str]:
        """클러스터링으로 대표적인 댓글들 선택"""
        from sklearn.cluster import KMeans
        import numpy as np
        
        if len(comments) < 10:
            return comments
        
        # 임베딩 생성
        embeddings = self.embedding_model.encode(comments)
        
        # 클러스터 개수 동적 결정 (댓글 수에 따라)
        n_clusters = min(8, max(3, len(comments) // 20))
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 각 클러스터에서 중심에 가장 가까운 댓글 선택
        representative = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) == 0:
                continue
            
            cluster_embeddings = embeddings[cluster_indices]
            cluster_center = kmeans.cluster_centers_[i]
            
            # 중심과 가장 가까운 댓글 찾기
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            representative.append(comments[closest_idx])
        
        return representative
    
    def categorize_comments(self, comments: List[str], metadatas: List[Dict]) -> Dict:
        """댓글을 특성별로 분류"""
        import re
        from datetime import datetime
        
        high_engagement = []  # 좋아요 많은 댓글
        questions = []        # 질문 댓글
        recent = []          # 최신 댓글
        total_likes = 0
        
        # 댓글과 메타데이터를 함께 처리
        comment_data = list(zip(comments, metadatas))
        
        # 좋아요 기준 정렬
        sorted_by_likes = sorted(comment_data, 
                               key=lambda x: x[1].get('likes_count', 0), 
                               reverse=True)
        
        # 시간 기준 정렬  
        sorted_by_time = sorted(comment_data,
                              key=lambda x: x[1].get('published_at', ''),
                              reverse=True)
        
        for comment, metadata in comment_data:
            likes_count = metadata.get('likes_count', 0)
            total_likes += likes_count
            
            # 좋아요 많은 댓글 (상위 30%)
            if likes_count > 0:
                high_engagement.append(comment)
            
            # 질문 패턴 감지
            if any(pattern in comment for pattern in ['?', '？', '궁금', '문의', '질문', '어떻게', '왜', '언제']):
                questions.append(comment)
        
        # 최신 댓글 (상위 30%)
        recent = [comment for comment, _ in sorted_by_time[:len(comments)//3]]
        
        return {
            'high_engagement': high_engagement[:15],
            'questions': questions[:10], 
            'recent': recent[:15],
            'avg_likes': total_likes / len(comments) if comments else 0
        }
    
    def analyze_temporal_trends(self, comments: List[str], metadatas: List[Dict]) -> Dict:
        """시간대별 활동 패턴 분석"""
        from collections import Counter
        from datetime import datetime
        
        hours = []
        dates = []
        
        for metadata in metadatas:
            published_at = metadata.get('published_at', '')
            if published_at:
                try:
                    dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    hours.append(dt.hour)
                    dates.append(dt.date())
                except:
                    continue
        
        if not hours:
            return {'peak_hours': '정보 없음', 'date_range': '정보 없음'}
        
        # 가장 활발한 시간대 찾기
        hour_counts = Counter(hours)
        peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else 0
        
        # 활동 기간
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            date_range = f"{min_date} ~ {max_date}"
        else:
            date_range = "정보 없음"
        
        return {
            'peak_hours': f"{peak_hour}시경",
            'date_range': date_range
        }

# 사용 예시
def main():
    # 설정
    S3_BUCKET = "your-vectordb-bucket"
    OPENAI_API_KEY = "your-openai-api-key"
    
    # 매니저 초기화
    manager = InstagramVectorDBManager(S3_BUCKET, OPENAI_API_KEY)
    
    # 예시 JSON 데이터
    sample_data = {
        "accountId": 1,
        "contentId": 76,
        "contentUrl": "https://www.instagram.com/p/DMohJuePSL4/",
        "commentsCount": 2,
        "crawledAt": "2025-09-21T11:35:01.609756900Z",
        "comments": [
            {
                "accountNickname": "well.freedom",
                "externalCommentId": "18066047266946777",
                "text": "오 완전 유용할거같아용 🔥🔥",
                "likesCount": 2,
                "publishedAt": "2025-07-28T01:41:12.000Z"
            },
            {
                "accountNickname": "kunst_gyun",
                "externalCommentId": "18061851164464824",
                "text": "만두님, 안녕하세요!\n혹시 이메일 한번 확인 가능하실까요? @tiro.kr 관련 건으로 연락드렸습니다!",
                "likesCount": 0,
                "publishedAt": "2025-08-08T06:02:00.000Z"
            }
        ]
    }
    
    # 처리 실행
    summary = manager.process_instagram_data(sample_data)
    print("\n" + "="*50)
    print("FINAL SUMMARY:")
    print(summary)

if __name__ == "__main__":
    main()
